# CaptionerSDAM_crop_opt_ordered.py
import torch
from PIL import Image
import numpy as np
from typing import List, Tuple
from transformers import SamModel, SamProcessor, AutoModel
import gc
import time
import concurrent.futures
from functools import partial
import threading

class DAMBatchProcessor:
    def __init__(self, dam_model, max_workers=4):
        self.dam = dam_model
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
    
    def process_batch_ordered(self, crops, masks, prompt, params):
        """Process multiple (crop, mask) pairs in parallel but maintain order"""
        results = [None] * len(crops)
        
        def process_and_store(index, crop, mask):
            try:
                mask_img = Image.fromarray((mask * 255).astype(np.uint8)) if isinstance(mask, np.ndarray) else mask
                description = ''.join(self.dam.get_description(
                    crop, mask_img,
                    prompt,
                    streaming=True,
                    **params
                ))
                return index, description.strip()
            except Exception as e:
                print(f"Error in DAM processing for index {index}: {e}")
                return index, f"[Error: {str(e)}]"
        
        # Submit all tasks
        futures = []
        for i, (crop, mask) in enumerate(zip(crops, masks)):
            future = self.executor.submit(process_and_store, i, crop, mask)
            futures.append(future)
        
        # Collect results maintaining order
        for future in concurrent.futures.as_completed(futures):
            index, result = future.result()
            results[index] = result
        
        return results
    
    def shutdown(self):
        self.executor.shutdown()

class CaptionerSDAM_crop_opt2:
    """
    Captioner basato su crop con ordinamento garantito
    """

    DEFAULT_PROMPT = "<image>\nGive a short description of the masked area."

    PRESET_PARAMS = {
        'balanced': {'temperature': 0.6, 'top_p': 0.75, 'max_new_tokens': 64}
    }

    def __init__(self, device=None, prompt=None, sam_batch_size=1, dam_workers=2):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DEFAULT_PROMPT = prompt or self.DEFAULT_PROMPT
        self.sam_batch_size = sam_batch_size
        self.dam_workers = min(dam_workers, 2)  # Start conservatively
        self._initialize_models()

    def _initialize_models(self):
        """Carica SAM e DAM."""
        print("Caricamento modelli...")

        # SAM
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
        
        # DAM Batch Processor
        self.dam_processor = DAMBatchProcessor(self.dam, max_workers=self.dam_workers)

        print(f"Modelli caricati! SAM batch: {self.sam_batch_size}, DAM workers: {self.dam_workers}")

    def _generate_fixed_points(self, width: int, height: int) -> List[List[int]]:
        """5 punti proporzionali: centro + 4 quadranti"""
        return [
            [width // 2, height // 2],
            [width // 4, height // 4],
            [3 * width // 4, height // 4],
            [width // 4, 3 * height // 4],
            [3 * width // 4, 3 * height // 4],
        ]

    def _crop_around_point(self, image: Image.Image, point: List[int], crop_ratio: float = 0.25) -> Tuple[Image.Image, List[int], Tuple[int,int]]:
        """Ritaglia un crop centrato sul punto"""
        w, h = image.size
        cx, cy = point
        
        base_size = min(w, h)
        crop_size_pixels = int(base_size * crop_ratio)
        half = crop_size_pixels // 2

        left = max(0, cx - half)
        top = max(0, cy - half)
        right = min(w, cx + half)
        bottom = min(h, cy + half)

        crop = image.crop((left, top, right, bottom))
        local_x = cx - left
        local_y = cy - top

        return crop, [local_x, local_y], (left, top)

    @torch.no_grad()
    def _apply_sam(self, crop: Image.Image, local_point: List[int]) -> np.ndarray:
        """Ottiene maschera da SAM (versione singola)"""
        try:
            inputs = self.sam_processor(crop, return_tensors="pt",
                                        input_points=[[local_point]], input_labels=[[1]])

            device_inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            outputs = self.sam_model(**device_inputs)

            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                device_inputs["original_sizes"].cpu(),
                device_inputs["reshaped_input_sizes"].cpu()
            )[0][0]

            best_mask_idx = int(outputs.iou_scores[0, 0].argmax().item())
            return masks[best_mask_idx].numpy()

        except Exception as e:
            print(f"Errore in _apply_sam: {e}")
            return np.zeros((crop.height, crop.width), dtype=np.uint8)

    def generate_descriptions(self, images: List[Image.Image]) -> List[List[str]]:
        """
        Genera 5 descrizioni per ogni immagine mantenendo l'ordine corretto
        """
        all_results = []

        for image_idx, image in enumerate(images):
            width, height = image.size
            points = self._generate_fixed_points(width, height)

            crops = []
            local_points_list = []
            
            # Preparazione crops
            for pt in points:
                crop, local_point, _ = self._crop_around_point(image, pt, crop_ratio=0.25)
                crops.append(crop)
                local_points_list.append(local_point)

            # SAM processing
            masks = []
            for i in range(0, len(crops), self.sam_batch_size):
                batch_crops = crops[i:i + self.sam_batch_size]
                batch_points = local_points_list[i:i + self.sam_batch_size]
                
                # Process SAM sequentially to ensure order
                for crop, point in zip(batch_crops, batch_points):
                    mask = self._apply_sam(crop, point)
                    masks.append(mask)

            # DAM processing with guaranteed order
            descriptions = self.dam_processor.process_batch_ordered(
                crops, masks, self.DEFAULT_PROMPT, self.PRESET_PARAMS['balanced']
            )

            # Verify order
            if len(descriptions) != len(points):
                print(f"⚠️  Attenzione: {len(descriptions)} descrizioni invece di {len(points)} per immagine {image_idx}")
                # Fallback: genera descrizioni vuote per mantenere l'ordine
                descriptions = descriptions[:len(points)]
                while len(descriptions) < len(points):
                    descriptions.append("[Error: Missing description]")

            all_results.append(descriptions)

            if (image_idx + 1) % 10 == 0:
                self.cleanup_memory()

        return all_results

    def cleanup_memory(self):
        """Libera memoria GPU/CPU."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    def shutdown(self):
        """Chiudi tutti i processi"""
        self.dam_processor.shutdown()
        self.cleanup_memory()