# CaptionerSDAM - versione snellita con 5 punti fissi
import torch
from PIL import Image
import numpy as np
from typing import List
from transformers import SamModel, SamProcessor, AutoModel
import gc

class CaptionerSDAM_fixed:
    """
    Captioner ottimizzato:
    - Usa 5 punti fissi proporzionali (centro + 4 quadranti)
    - Un solo preset "balanced"
    - Prompt breve per descrizioni concise
    """
    DEFAULT_PROMPT = "<image>\nGive a short description of the masked area." # "<image>\nSummarize the masked region." o "<image>\nGive a short description of the masked area." o "<image>\nDescribe the masked region briefly"

    PRESET_PARAMS = {
        'balanced': {'temperature': 0.6, 'top_p': 0.75, 'max_new_tokens': 64}
    }

    def __init__(self, device=None, prompt=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DEFAULT_PROMPT = prompt or "<image>\nGive a short description of the masked area."
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

        print("Modelli caricati con successo!")

    def _generate_fixed_points(self, width: int, height: int) -> List[List[int]]:
        """5 punti proporzionali: centro + 4 quadranti"""
        return [
            [width // 2, height // 2],                # centro
            [width // 4, height // 4],                # top-left
            [3 * width // 4, height // 4],            # top-right
            [width // 4, 3 * height // 4],            # bottom-left
            [3 * width // 4, 3 * height // 4],        # bottom-right
        ]

    @torch.no_grad()
    def _apply_sam(self, image, **kwargs):
        """Ottiene la maschera da SAM dato un set di punti."""
        try:
            inputs = self.sam_processor(image, return_tensors="pt", **kwargs)

            # Sposta gli input sul device
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
            return np.zeros((image.height, image.width), dtype=np.uint8)

    def _generate_description(self, image: Image.Image, mask: np.ndarray) -> str:
        """Genera descrizione con DAM (balanced)."""
        try:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            description = ''.join(self.dam.get_description(
                image, mask_img,
                self.DEFAULT_PROMPT, # si passa il prompt precedentemente parametrizzato
                streaming=True,
                **self.PRESET_PARAMS['balanced']
            ))
            return description.strip()
        except Exception as e:
            return f"[Errore generazione descrizione: {e}]"

    def generate_descriptions(self, images: List[Image.Image]) -> List[List[str]]:
        """
        Genera 5 descrizioni per ogni immagine (una per punto fisso).
        Ritorna: lista di liste di caption.
        """
        all_results = []

        for image in images:
            width, height = image.size
            points = self._generate_fixed_points(width, height)

            captions = []
            for pt in points:
                mask = self._apply_sam(image, input_points=[[pt]], input_labels=[[1]])
                desc = self._generate_description(image, mask)
                captions.append(desc)

            all_results.append(captions)

        return all_results

    def cleanup_memory(self):
        """Libera memoria GPU/CPU."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
