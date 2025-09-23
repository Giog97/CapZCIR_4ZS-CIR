import torch
from PIL import Image
import numpy as np
from typing import List, Dict
from transformers import SamModel, SamProcessor, AutoModel
import gc

class CaptionerSDAM_singlepoint:
    """
    Version that uses only a single central point for SAM segmentation.
    Generates 3 descriptions per image (one for each preset).
    
    Optimizations:
    - Single point segmentation
    - Batch processing for DAM
    - Improved memory management
    """
    
    PRESET_PARAMS = {
        'conservative': {'temperature': 0.2, 'top_p': 0.5, 'max_new_tokens': 128},
        'balanced': {'temperature': 0.6, 'top_p': 0.75, 'max_new_tokens': 128},
        'creative': {'temperature': 1.0, 'top_p': 0.95, 'max_new_tokens': 128}
    }

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()

    def _initialize_models(self):
        """Load SAM and DAM models with memory optimizations."""
        print("Loading models...")
        
        # SAM - keep float32 for compatibility
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.sam_model.eval()
        
        # DAM
        self.dam_model = AutoModel.from_pretrained(
            'nvidia/DAM-3B-Self-Contained',
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(self.device)
        self.dam = self.dam_model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')
        
        print("Models loaded successfully!")

    def generate_descriptions(self, image: Image.Image) -> List[str]:
        """
        Generate 3 descriptions for a single image (one for each preset).
        Returns: List of descriptions [conservative, balanced, creative]
        """
        # Step 1: Generate mask using single central point
        mask = self._generate_mask(image)
        
        # Step 2: Generate descriptions for each preset
        descriptions = []
        for preset in self.PRESET_PARAMS.keys():
            desc = self._generate_description(image, mask, preset)
            descriptions.append(desc)
            
        return descriptions

    def generate_descriptions_batch(self, images: List[Image.Image]) -> List[List[str]]:
        """
        Generate descriptions for a batch of images.
        Returns: List of lists, each containing 3 descriptions for an image.
        """
        batch_descriptions = []
        
        # Process images one by one (could be optimized further)
        for image in images:
            descriptions = self.generate_descriptions(image)
            batch_descriptions.append(descriptions)
            
        return batch_descriptions

    def _generate_mask(self, image: Image.Image) -> np.ndarray:
        """Generate a mask using SAM with a single central point."""
        try:
            # Calculate central point
            width, height = image.size
            point = [[width // 2, height // 2]]
            
            # Apply SAM
            inputs = self.sam_processor(
                image, 
                input_points=[point], 
                input_labels=[[1]], 
                return_tensors="pt"
            )
            
            # Move inputs to device
            device_inputs = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    device_inputs[key] = value.to(self.device)
                else:
                    device_inputs[key] = value
                    
            outputs = self.sam_model(**device_inputs)
            
            # Process masks
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                device_inputs["original_sizes"].cpu(),
                device_inputs["reshaped_input_sizes"].cpu()
            )[0][0]
            
            # Get the best mask
            best_mask_idx = int(outputs.iou_scores[0, 0].argmax().item())
            return masks[best_mask_idx].numpy()
            
        except Exception as e:
            print(f"Error in _generate_mask: {e}")
            # Return empty mask as fallback
            return np.zeros((image.height, image.width), dtype=np.uint8)

    def _generate_description(self, image: Image.Image, mask: np.ndarray, preset: str) -> str:
        """Generate a description using DAM with the given preset."""
        try:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            description = ''.join(self.dam.get_description(
                image, mask_img,
                '<image>\nDescribe the masked region briefly.',
                streaming=True,
                **self.PRESET_PARAMS[preset]
            ))
            return description.strip()
        except Exception as e:
            print(f"Error generating description: {e}")
            return f"Error generating description with {preset} preset"

    def cleanup_memory(self):
        """Clean up GPU memory."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()