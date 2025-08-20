# CaptionerSDAM ottimizzato
import torch
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
from transformers import SamModel, SamProcessor, AutoModel
import gc

class CaptionerSDAM_opt:
    """
    Versione ottimizzata per il batch processing.
    Ottimizzazioni principali:
    - Batch processing per SAM e DAM
    - Riuso delle maschere per diversi preset
    - Gestione memoria migliorata
    - Cache dei punti griglia
    """
    
    PRESET_PARAMS = {
        'conservative': {'temperature': 0.2, 'top_p': 0.5, 'max_new_tokens': 128},  # Ridotto ulteriormente
        'balanced': {'temperature': 0.6, 'top_p': 0.75, 'max_new_tokens': 128},
        'creative': {'temperature': 1.0, 'top_p': 0.95, 'max_new_tokens': 128}
    }

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()
        # Cache per i punti griglia
        self._grid_points_cache = {}

    def _initialize_models(self):
        """Load SAM and DAM models con ottimizzazioni memoria."""
        print("Caricamento modelli...")
        
        # SAM - mantieni float32 per evitare problemi di compatibilità
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base") # facebook/sam-vit-base o facebook/sam-vit-huge
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device) # facebook/sam-vit-base o facebook/sam-vit-huge
        self.sam_model.eval()  # Modalità evaluation
        
        # DAM
        self.dam_model = AutoModel.from_pretrained(
            'nvidia/DAM-3B-Self-Contained',
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(self.device)
        self.dam = self.dam_model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')
        
        print("Modelli caricati con successo!")

    def generate_descriptions_batch(
        self, 
        images: List[Image.Image], 
        max_level: int = 4
    ) -> List[Dict[int, List[str]]]:
        """
        Processa un batch di immagini contemporaneamente.
        Returns: Lista di dizionari {level: [desc_conservative, desc_balanced, desc_creative]}
        """
        batch_results = []
        
        # Processa tutte le immagini per ogni livello per ottimizzare l'uso della GPU
        for level in range(max_level + 1):
            level_results = self._process_level_batch(images, level)
            
            # Organizza i risultati per immagine
            if level == 0:
                batch_results = [{level: result} for result in level_results]
            else:
                for i, result in enumerate(level_results):
                    batch_results[i][level] = result
                    
        return batch_results

    def _process_level_batch(self, images: List[Image.Image], level: int) -> List[List[str]]:
        """Processa un batch di immagini per un singolo livello."""
        # Step 1: Genera le maschere per tutte le immagini del batch
        masks = self._generate_masks_batch(images, level)
        
        # Step 2: Genera le descrizioni per ogni preset
        level_results = []
        for i, (image, mask) in enumerate(zip(images, masks)):
            descriptions = []
            for preset in self.PRESET_PARAMS.keys():
                desc = self._generate_description(image, mask, preset)
                descriptions.append(desc)
            level_results.append(descriptions)
            
        return level_results

    def _generate_masks_batch(self, images: List[Image.Image], level: int) -> List[np.ndarray]:
        """Genera maschere per un batch di immagini usando SAM."""
        masks = []
        
        # Processa le immagini in mini-batch per gestire la memoria
        mini_batch_size = min(4, len(images))  # Batch size ridotto per SAM
        
        for i in range(0, len(images), mini_batch_size):
            mini_batch = images[i:i + mini_batch_size]
            mini_masks = self._process_sam_mini_batch(mini_batch, level)
            masks.extend(mini_masks)
            
            # Pulizia memoria dopo ogni mini-batch
            torch.cuda.empty_cache() if self.device.type == 'cuda' else None
            
        return masks

    def _process_sam_mini_batch(self, images: List[Image.Image], level: int) -> List[np.ndarray]:
        """Processa un mini-batch con SAM."""
        mini_masks = []
        
        for image in images:
            # Usa cache per i punti griglia
            cache_key = (image.width, image.height, level)
            if cache_key not in self._grid_points_cache:
                self._grid_points_cache[cache_key] = self._generate_grid_points(
                    image.width, image.height, level
                )
            points = self._grid_points_cache[cache_key]
            
            # Applica SAM
            mask = self._apply_sam(image, input_points=[points], input_labels=[[1] * len(points)])
            mini_masks.append(mask)
            
        return mini_masks

    def _generate_grid_points(self, width: int, height: int, level: int) -> List[List[int]]:
        """Generate grid points for a given level (cached)."""
        if level == 0:
            return [[width // 2, height // 2]]
        grid_size = level + 2
        return [
            [width * i // grid_size, height * j // grid_size]
            for i in range(1, grid_size)
            for j in range(1, grid_size)
        ]

    @torch.no_grad()  # Disabilita gradient computation
    def _apply_sam(self, image, **kwargs):
        """Get a single mask from SAM using all input points."""
        try:
            inputs = self.sam_processor(image, return_tensors="pt", **kwargs)
            
            # Sposta tutti gli input su device in modo sicuro
            device_inputs = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    device_inputs[key] = value.to(self.device)
                else:
                    device_inputs[key] = value
                    
            outputs = self.sam_model(**device_inputs)
            
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                device_inputs["original_sizes"].cpu(),
                device_inputs["reshaped_input_sizes"].cpu()
            )[0][0]
            
            # Assicurati che l'indice sia un int
            best_mask_idx = int(outputs.iou_scores[0, 0].argmax().item())
            return masks[best_mask_idx].numpy()
            
        except Exception as e:
            print(f"Errore in _apply_sam: {e}")
            # Ritorna una maschera vuota come fallback
            return np.zeros((image.height, image.width), dtype=np.uint8)

    def _generate_description(self, image: Image.Image, mask: np.ndarray, preset: str) -> str:
        """Generate a description using DAM with the given preset."""
        try:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            description = ''.join(self.dam.get_description(
                image, mask_img,
                '<image>\nDescribe the masked region briefly.',  # Prompt più breve di: '<image>\nDescribe the masked region in detail'
                streaming=True,
                **self.PRESET_PARAMS[preset]
            ))
            return description.strip()
        except Exception as e:
            print(f"Errore nella generazione descrizione: {e}")
            return f"Error generating description with {preset} preset"

    def generate_all_descriptions_batch(self, images: List[Image.Image]) -> List[List[str]]:
        """
        Genera 15 descrizioni per ogni immagine nel batch.
        Returns: Lista di liste, una per ogni immagine.
        """
        batch_descriptions = []
        descriptions_dicts = self.generate_descriptions_batch(images, max_level=4)
        
        for descriptions_dict in descriptions_dicts:
            all_descriptions = []
            for level in sorted(descriptions_dict.keys()):
                all_descriptions.extend(descriptions_dict[level])
            batch_descriptions.append(all_descriptions)
            
        return batch_descriptions

    def cleanup_memory(self):
        """Pulisce la memoria GPU."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

