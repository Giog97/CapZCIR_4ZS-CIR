# Script per elaborare le immagini mancanti che non sono state eleaborate con lo script principale
# Inutile se non riesci a ricaricare le immagini NON corrotte

import warnings
warnings.filterwarnings("ignore", message="`torch.utils._pytree._register_pytree_node` is deprecated")
import json
import torch
from pathlib import Path
from typing import List, Dict
from PIL import Image
from tqdm import tqdm
from model.captioner_sdam_opt import CaptionerSDAM_opt
import time
import gc
from concurrent.futures import ThreadPoolExecutor
import threading

# Configurazione
INPUT_JSON = Path('./data/files/laion_combined_info.json')
IMAGE_DIR = Path('./data/datasets/laion_cir_combined')
OUTPUT_JSON = Path('./data/files/laion_combined_dam_multi.json')
EXISTING_OUTPUT = Path('./data/files/laion_combined_dam_multi.json')  # File esistente
NUM_CAPTIONS = 15
BATCH_SIZE = 4
CHECKPOINT_INTERVAL = 50
MAX_WORKERS = 4

# Inizializza il captioner
print("Inizializzazione del captioner...")
captioner = CaptionerSDAM_opt()
results_lock = threading.Lock()

def find_missing_images():
    """Trova le immagini che non sono state processate correttamente."""
    try:
        # Carica i dati originali
        with open(INPUT_JSON, 'r') as f:
            original_data = json.load(f)
        
        # Carica i dati processati
        with open(EXISTING_OUTPUT, 'r') as f:
            processed_data = json.load(f)
        
        # Crea dizionari per accesso rapido
        processed_dict = {item["ref_image_id"]: item for item in processed_data}
        original_dict = {item["ref_image_id"]: item for item in original_data}
        
        # Trova immagini mancanti
        missing_items = []
        for ref_id, original_item in original_dict.items():
            if ref_id not in processed_dict:
                missing_items.append(original_item)
            else:
                # Controlla anche se il numero di caption è insufficiente
                processed_item = processed_dict[ref_id]
                if ("multi_caption_dam" not in processed_item or 
                    len(processed_item.get("multi_caption_dam", [])) < NUM_CAPTIONS):
                    missing_items.append(original_item)
        
        return missing_items
    
    except Exception as e:
        print(f"Errore nel trovare immagini mancanti: {e}")
        return []

def load_images_parallel(image_ids: List[str]) -> List[Image.Image]:
    """Carica le immagini in parallelo."""
    def load_single_image(image_id: str) -> Image.Image:
        try:
            image_path = IMAGE_DIR / f"{image_id.zfill(7)}.png"
            image = Image.open(image_path).convert('RGB')
            image.load()
            return image
        except Exception as e:
            print(f"Errore caricamento immagine {image_id}: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        images = list(executor.map(load_single_image, image_ids))
    
    valid_pairs = [(img, img_id) for img, img_id in zip(images, image_ids) if img is not None]
    if not valid_pairs:
        return [], []
    
    valid_images, valid_ids = zip(*valid_pairs)
    return list(valid_images), list(valid_ids)

def process_batch(batch: List[dict]) -> List[dict]:
    """Processa un batch di elementi."""
    batch_start_time = time.time()
    try:
        image_ids = [item["ref_image_id"] for item in batch]
        
        print(f"🔄 Processando batch con {len(image_ids)} immagini...")
        
        # Carica immagini
        load_start = time.time()
        images, valid_ids = load_images_parallel(image_ids)
        load_time = time.time() - load_start
        
        if not images:
            print(f"❌ Nessuna immagine valida nel batch")
            return []
        
        print(f"✅ Caricate {len(images)}/{len(image_ids)} immagini in {load_time:.2f}s")
        
        # Genera descrizioni
        gen_start = time.time()
        batch_descriptions = captioner.generate_all_descriptions_batch(images)
        gen_time = time.time() - gen_start
        
        print(f"📝 Generazione descrizioni completata in {gen_time:.2f}s")
        
        # Costruisci risultati
        batch_results = []
        valid_items = [item for item in batch if item["ref_image_id"] in valid_ids]
        
        for item, descriptions in zip(valid_items, batch_descriptions):
            if descriptions and len(descriptions) >= NUM_CAPTIONS:
                batch_results.append({
                    "ref_image_id": item["ref_image_id"],
                    "relative_cap": item["relative_cap"],
                    "tgt_image_id": item["tgt_image_id"],
                    "multi_caption_dam": descriptions[:NUM_CAPTIONS]
                })
            else:
                desc_count = len(descriptions) if descriptions else 0
                print(f"⚠️  Solo {desc_count} descrizioni per {item['ref_image_id']}")
        
        batch_total_time = time.time() - batch_start_time
        success_rate = len(batch_results) / len(batch) * 100
        
        print(f"✨ Batch completato: {len(batch_results)}/{len(batch)} successi ({success_rate:.1f}%) in {batch_total_time:.2f}s")
        
        return batch_results
        
    except Exception as e:
        batch_error_time = time.time() - batch_start_time
        print(f"💥 Errore nel batch processing dopo {batch_error_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    # Trova immagini mancanti
    print("Ricerca immagini mancanti...")
    missing_data = find_missing_images()
    
    if not missing_data:
        print("🎉 Tutte le immagini sono già state processate correttamente!")
        return
    
    print(f"Trovate {len(missing_data)} immagini da rielaborare")
    
    # Carica i risultati esistenti
    try:
        with open(EXISTING_OUTPUT, 'r') as f:
            existing_results = json.load(f)
    except:
        existing_results = []
    
    # Processa solo le immagini mancanti
    total_batches = (len(missing_data) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()
    
    print(f"\n🚀 AVVIO RI-ELABORAZIONE")
    print(f"{'='*60}")
    print(f"📁 Immagini da rielaborare: {len(missing_data)}")
    print(f"📦 Batch totali: {total_batches} da {BATCH_SIZE} immagini")
    print(f"{'='*60}")
    
    results = []
    progress_bar = tqdm(range(total_batches), desc="Batch processati")
    
    for batch_idx in progress_bar:
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(missing_data))
        batch = missing_data[start_idx:end_idx]
        
        # Processa il batch
        batch_results = process_batch(batch)
        
        with results_lock:
            results.extend(batch_results)
        
        # Aggiorna progress bar
        progress_bar.set_postfix({
            'success': f'{len(batch_results)}/{len(batch)}',
            'total': f'{len(results)}/{len(missing_data)}'
        })
        
        # Pulizia memoria periodica
        if (batch_idx + 1) % 50 == 0:
            captioner.cleanup_memory()
            gc.collect()
    
    # Combina i risultati esistenti con quelli nuovi
    final_results = existing_results + results
    
    # Salva i risultati finali
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Statistiche finali
    total_time = time.time() - start_time
    print(f"\n🎉 RI-ELABORAZIONE COMPLETATA!")
    print(f"⏱️  Tempo totale: {total_time/60:.1f} minuti")
    print(f"✅ Immagini rielaborate: {len(results)}/{len(missing_data)}")
    print(f"📊 Totale finale: {len(final_results)}/{len(missing_data) + len(existing_results)}")

if __name__ == "__main__":
    main()