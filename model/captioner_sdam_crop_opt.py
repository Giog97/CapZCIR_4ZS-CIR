# CaptionerSDAM - versione con crop attorno ai punti fissi e SAM batch --> captioner più veloce perchè gestisce SAM per 1 immagine in parallelo
import torch
from PIL import Image
import numpy as np
from typing import List, Tuple
from transformers import SamModel, SamProcessor, AutoModel
import gc
import time

# Ottimizzazioni fatte:
# - Per ora aggiunto solo SAM procesing con Batch (quindi passo più crop e più punti a SAM in modo da parallelizzare)
# Da testare

class CaptionerSDAM_crop_opt:
    """
    Captioner basato su crop:
    - 5 punti fissi proporzionali (centro + 4 quadranti)
    - Per ogni punto: crop locale + SAM → mask + DAM → descrizione
    - DAM riceve direttamente (crop + mask), non l'immagine intera
    - SAM processato in batch per ottimizzazione
    """

    DEFAULT_PROMPT = "<image>\nGive a short description of the masked area."

    PRESET_PARAMS = {
        'balanced': {'temperature': 0.6, 'top_p': 0.75, 'max_new_tokens': 64}
    }

    def __init__(self, device=None, prompt=None, sam_batch_size=4):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DEFAULT_PROMPT = prompt or self.DEFAULT_PROMPT
        self.sam_batch_size = sam_batch_size  # Nuovo parametro per batch size SAM
        self._initialize_models()

    def _initialize_models(self):
        """Carica SAM e DAM."""
        print("Caricamento modelli...")

        # SAM (base per velocità)
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

        print(f"Modelli caricati con successo! Batch size SAM: {self.sam_batch_size}")

    def _generate_fixed_points(self, width: int, height: int) -> List[List[int]]:
        """5 punti proporzionali: centro + 4 quadranti"""
        return [
            [width // 2, height // 2],                # centro
            [width // 4, height // 4],                # top-left
            [3 * width // 4, height // 4],            # top-right
            [width // 4, 3 * height // 4],            # bottom-left
            [3 * width // 4, 3 * height // 4],        # bottom-right
        ]

    def _crop_around_point(self, image: Image.Image, point: List[int], crop_ratio: float = 0.25) -> Tuple[Image.Image, List[int], Tuple[int,int]]:
        """
        Ritaglia un crop centrato sul punto, proporzionale all'immagine originale.
        
        Args:
            image: L'immagine PIL originale.
            point: Il punto [x, y] attorno a cui ritagliare.
            crop_ratio: La frazione (es. 0.2 per 20%) della dimensione *minore* dell'immagine da usare come lato del crop.
                        Un valore più alto = un'area di crop più grande.

        Returns:
            crop: L'immagine ritagliata.
            local_point: Le coordinate del punto originale all'interno del crop.
            offset: L'offset (left, top) del crop nell'immagine originale.
        """
        w, h = image.size
        cx, cy = point
        
        # Calcola la dimensione del crop come frazione della dimensione minore
        base_size = min(w, h)
        crop_size_pixels = int(base_size * crop_ratio)
        
        half = crop_size_pixels // 2

        # bounding box del crop (stessa logica per gestire i bordi)
        left = max(0, cx - half)
        top = max(0, cy - half)
        right = min(w, cx + half)
        bottom = min(h, cy + half)

        crop = image.crop((left, top, right, bottom))

        # punto traslato nel crop
        local_x = cx - left
        local_y = cy - top

        return crop, [local_x, local_y], (left, top)

    @torch.no_grad()
    def _apply_sam(self, crop: Image.Image, local_point: List[int]) -> np.ndarray:
        """Ottiene maschera da SAM sul crop attorno al punto (versione singola)."""
        try:
            inputs = self.sam_processor(crop, return_tensors="pt",
                                        input_points=[[local_point]], input_labels=[[1]])

            device_inputs = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }

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

    @torch.no_grad()
    def _apply_sam_batch(self, crops: List[Image.Image], local_points: List[List[int]]) -> List[np.ndarray]:
        """Ottiene maschere da SAM processando multiple crops in batch."""
        try:
            # Prepara input points e labels per il batch
            input_points = [[point] for point in local_points]
            input_labels = [[1] for _ in local_points]
            
            # Processa il batch con SAM
            inputs = self.sam_processor(
                crops, 
                return_tensors="pt",
                input_points=input_points,
                input_labels=input_labels
            )

            device_inputs = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }

            outputs = self.sam_model(**device_inputs)

            # Post-process delle maschere
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                device_inputs["original_sizes"].cpu(),
                device_inputs["reshaped_input_sizes"].cpu()
            )
            
            # Estrai la migliore maschera per ogni input
            batch_masks = []
            for i in range(len(crops)):
                best_mask_idx = int(outputs.iou_scores[i, 0].argmax().item())
                mask = masks[i][0][best_mask_idx].numpy()
                batch_masks.append(mask)
                
            return batch_masks

        except Exception as e:
            print(f"Errore in _apply_sam_batch: {e}")
            return [np.zeros((crop.height, crop.width), dtype=np.uint8) for crop in crops]

    def _generate_description(self, crop: Image.Image, mask: np.ndarray) -> str:
        """Genera descrizione con DAM (balanced) usando crop + mask."""
        try:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            description = ''.join(self.dam.get_description(
                crop, mask_img,
                self.DEFAULT_PROMPT,
                streaming=True, # se voglio usare il batching questo dovrebbe essere impostato a FALSE? # Da verificare
                **self.PRESET_PARAMS['balanced']
            ))
            return description.strip()
        except Exception as e:
            return f"[Errore generazione descrizione: {e}]"

    def generate_descriptions(self, images: List[Image.Image], crop_size: int = 256) -> List[List[str]]:
        """
        Genera 5 descrizioni per ogni immagine (una per punto fisso).
        Strategia: crop centrato su ciascun punto -> SAM (in batch) -> DAM.
        """
        all_results = []

        for image in images:
            width, height = image.size
            points = self._generate_fixed_points(width, height)

            captions = []
            crops = []
            local_points_list = []
            
            # Fase 1: Preparazione di tutti i crops
            start_time_crop = time.time()
            for pt in points:
                crop, local_point, _ = self._crop_around_point(image, pt, crop_ratio=0.25)
                crops.append(crop)
                local_points_list.append(local_point)
            crop_time = time.time() - start_time_crop

            # Fase 2: Processamento SAM in batch
            start_time_sam = time.time()
            masks = []
            for i in range(0, len(crops), self.sam_batch_size):
                batch_crops = crops[i:i + self.sam_batch_size]
                batch_points = local_points_list[i:i + self.sam_batch_size]
                batch_masks = self._apply_sam_batch(batch_crops, batch_points)
                masks.extend(batch_masks)
            sam_time = time.time() - start_time_sam

            # Fase 3: Processamento DAM (ancora sequenziale per ora)
            dam_times = []
            for crop, mask in zip(crops, masks):
                start_time_dam = time.time()
                desc = self._generate_description(crop, mask)
                dam_time = time.time() - start_time_dam
                dam_times.append(dam_time)
                captions.append(desc)
                
                # Debug info per ogni punto
                pt = points[len(captions) - 1]
                #print(f"[DEBUG] Image{image}, Point {pt}: Crop={crop_time/5:.3f}s, SAM={sam_time/5:.3f}s, DAM={dam_time:.3f}s")

            all_results.append(captions)

        return all_results

    def cleanup_memory(self):
        """Libera memoria GPU/CPU."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()