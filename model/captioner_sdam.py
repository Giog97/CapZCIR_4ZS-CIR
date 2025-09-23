import torch
from PIL import Image
import numpy as np
from typing import Dict, List
from transformers import SamModel, SamProcessor, AutoModel

class CaptionerSDAM:
    """
    Generates 3 descriptions per level (0 to max_level) using:
    - SAM: Single mask per level (all grid points passed at once).
    - DAM: 3 presets (conservative/balanced/creative) per mask.
    Output: {level: [desc_conservative, desc_balanced, desc_creative]}
    """
    
    PRESET_PARAMS = {
        'conservative': {'temperature': 0.2, 'top_p': 0.5, 'max_new_tokens': 256}, # modificato da 512 a 256 sennò avevo problemi di memoria
        'balanced': {'temperature': 0.6, 'top_p': 0.75, 'max_new_tokens': 256},
        'creative': {'temperature': 1.0, 'top_p': 0.95, 'max_new_tokens': 256}
    }

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()

    def _initialize_models(self):
        """Load SAM and DAM models."""
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        self.dam_model = AutoModel.from_pretrained(
            'nvidia/DAM-3B-Self-Contained',
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(self.device)
        self.dam = self.dam_model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')

    def generate_descriptions(
        self, 
        image: Image.Image, 
        max_level: int = 4  # Levels 0-4 → 15 descriptions (3 per level)
    ) -> Dict[int, List[str]]:
        """
        Returns: {level: [desc_conservative, desc_balanced, desc_creative]}
        """
        descriptions = {}
        for level in range(max_level + 1):
            # Step 1: Generate grid points for this level
            points = self._generate_grid_points(image.width, image.height, level)
            
            # Step 2: Pass ALL points to SAM to get a single combined mask
            mask = self._apply_sam(image, input_points=[points], input_labels=[[1] * len(points)])
            
            # Step 3: Generate 3 descriptions (one per preset) for this mask
            level_descriptions = [
                self._generate_description(image, mask, preset)
                for preset in self.PRESET_PARAMS.keys()
            ]
            descriptions[level] = level_descriptions
        return descriptions

    def _generate_grid_points(self, width: int, height: int, level: int) -> List[List[int]]:
        """Generate grid points for a given level (e.g., level 2 → 3x3 grid)."""
        if level == 0:
            return [[width // 2, height // 2]]  # 1 point at center
        grid_size = level + 2  # e.g., level 2 → 3x3 grid (9 points)
        return [
            [width * i // grid_size, height * j // grid_size]
            for i in range(1, grid_size)
            for j in range(1, grid_size)
        ]

    def _apply_sam(self, image, **kwargs):
        """Get a single mask from SAM using all input points."""
        inputs = self.sam_processor(image, return_tensors="pt", **kwargs).to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0][0]
        return masks[outputs.iou_scores[0, 0].argmax()].numpy()  # Best mask

    def _generate_description(self, image: Image.Image, mask: np.ndarray, preset: str) -> str:
        """Generate a description using DAM with the given preset."""
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        return ''.join(self.dam.get_description(
            image, mask_img,
            '<image>\nDescribe the masked region in detail.',
            streaming=True,
            **self.PRESET_PARAMS[preset]
        ))
    
    # Funzione che serve per generare le descriptions delle immagini dei dei dataset classe CaptionerSDAM
    def generate_all_descriptions(self, image: Image.Image) -> List[str]:
        """
        Generates 15 descriptions (3 per level × 5 levels) and returns them as a single list.
        """
        all_descriptions = []
        descriptions_dict = self.generate_descriptions(image, max_level=4)
        
        for level in sorted(descriptions_dict.keys()):
            all_descriptions.extend(descriptions_dict[level])
        
        return all_descriptions
    