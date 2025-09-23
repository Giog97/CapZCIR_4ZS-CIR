"""
Captioner ottimizzato con crop:
    - Usa 5 punti fissi proporzionali (centro + 4 quadranti)
    - SAM lavora sul crop
    - DAM riceve crop + mask
    - Un solo preset "balanced"
    - Prompt breve per descrizioni concise
"""
import warnings
warnings.filterwarnings("ignore", message="`torch.utils._pytree._register_pytree_node` is deprecated")
import json
import torch
from pathlib import Path
from typing import List, Dict
from PIL import Image
from tqdm import tqdm
from model.captioner_sdam_crop import CaptionerSDAM_crop   # 🔄 usa la nuova classe
from model.captioner_sdam_crop_opt import CaptionerSDAM_crop_opt   # --> captioner più veloce perchè gestisce SAM per 1 immagine in parallelo
import time
import gc
from concurrent.futures import ThreadPoolExecutor
import threading

import os
print(f"Numero core logici: {os.cpu_count()}")  # numero core logici --> MAX_WORKERS: da mettere uguale o poco meno di os.cpu_count()

# Configurazione ottimizzata
INPUT_JSON = Path('./data/files/laion_combined_info.json')
IMAGE_DIR = Path('./data/datasets/laion_cir_combined')
OUTPUT_JSON = Path('./data/files/laion_combined_dam_multi_crop.json')         # 🔄 output aggiornato
CHECKPOINT_JSON = Path('./data/files/checkpoint_dam_multi_crop.json')         # 🔄 checkpoint aggiornato
#OUTPUT_JSON = Path('./data/files/laion_combined_dam_multi_crop_opt.json')         # 🔄 output aggiornato
#CHECKPOINT_JSON = Path('./data/files/checkpoint_dam_multi_crop_opt.json') 
NUM_CAPTIONS = 5
BATCH_SIZE = 8 # esp fatti con 8  
CHECKPOINT_INTERVAL = 2  
MAX_WORKERS = 16  

# Inizializza il captioner una sola volta
print("Inizializzazione del captioner...")  

## ATTENZIONE SE IL CODICE NON VA A CAUSA DELLA MAMORIA RIDURRE PRIMA QUESTO BATCH_SIZE_SAM
captioner = CaptionerSDAM_crop() # 🔄 nuovo captioner
#captioner = CaptionerSDAM_crop_opt(sam_batch_size=3)  # Todo prova a variare sam_batch_size 
# Per GPU attuale (10.75 GB) - usa batch_size=1 come fallback sicuro
# 8 e 4 e 2 come batch size è troppo per lechunk

# Lock per thread safety
results_lock = threading.Lock()

def load_images_parallel(image_ids: List[str]) -> List[Image.Image]:
    """Carica le immagini in parallelo usando ThreadPoolExecutor."""
    def load_single_image(image_id: str) -> Image.Image:
        try:
            image_path = IMAGE_DIR / f"{image_id.zfill(7)}.png"
            image = Image.open(image_path).convert('RGB')
            image.load()  # Verifica validità
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

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    else:
        return f"{minutes:02d}m {secs:02d}s"

def process_batch(batch: List[dict]) -> List[dict]:
    batch_start_time = time.time()
    try:
        image_ids = [item["ref_image_id"] for item in batch]
        print(f"🔄 Processando batch con {len(image_ids)} immagini...")

        images, valid_ids = load_images_parallel(image_ids)
        if not images:
            print("❌ Nessuna immagine valida nel batch")
            return []

        print(f"✅ Caricate {len(images)}/{len(image_ids)} immagini")

        # Usa il nuovo captioner (crop)
        batch_descriptions = captioner.generate_descriptions(images)

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
                print(f"⚠️ Solo {len(descriptions)} descrizioni per {item['ref_image_id']}")

        return batch_results

    except Exception as e:
        print(f"💥 Errore nel batch: {e}")
        import traceback; traceback.print_exc()
        return []


def save_checkpoint(results: List[dict], batch_idx: int):
    checkpoint_data = {
        "batch_processed": batch_idx,
        "results": results,
        "timestamp": time.time()
    }
    with open(CHECKPOINT_JSON, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"Checkpoint salvato dopo batch {batch_idx}")

def load_checkpoint() -> tuple:
    if CHECKPOINT_JSON.exists():
        try:
            with open(CHECKPOINT_JSON, 'r') as f:
                checkpoint = json.load(f)
            return checkpoint["batch_processed"], checkpoint["results"]
        except Exception as e:
            print(f"Errore caricamento checkpoint: {e}")
    return 0, []

def main():
    with open(INPUT_JSON, 'r') as f:
        template_data = json.load(f)
    
    start_batch, results = load_checkpoint()
    
    total_batches = (len(template_data) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()
    
    print(f"\n🚀 AVVIO ELABORAZIONE")
    print(f"{'='*60}")
    print(f"📁 Dataset: {len(template_data)} immagini totali")
    print(f"📦 Configurazione: {total_batches} batch da {BATCH_SIZE} immagini")
    print(f"💾 Checkpoint ogni {CHECKPOINT_INTERVAL} batch")
    print(f"🎯 Obiettivo: {NUM_CAPTIONS} descrizioni per immagine")
    print(f"⚙️  Device: {captioner.device}")
    
    if start_batch > 0:
        print(f"🔄 Ripresa dal batch {start_batch + 1}/{total_batches}")
        print(f"✅ Già processate: {len(results)} immagini")
    
    print(f"{'='*60}")
    print(f"⏰ Avvio alle: {time.strftime('%H:%M:%S del %d/%m/%Y')}")
    print(f"{'='*60}\n")
    
    progress_bar = tqdm(
        range(start_batch, total_batches), 
        desc="Batch processati",
        initial=start_batch,
        total=total_batches
    )
    
    for batch_idx in progress_bar:
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(template_data))
        batch = template_data[start_idx:end_idx]
        
        batch_start = time.time()
        batch_results = process_batch(batch)
        batch_time = time.time() - batch_start
        
        with results_lock:
            results.extend(batch_results)
        
        processed_items = len(results)
        elapsed = time.time() - start_time
        current_batch_time = time.time() - batch_start
        
        if hasattr(main, 'recent_times'):
            main.recent_times.append(current_batch_time)
            if len(main.recent_times) > 10:
                main.recent_times.pop(0)
        else:
            main.recent_times = [current_batch_time]
        
        avg_batch_time = sum(main.recent_times) / len(main.recent_times)
        items_per_batch = len(batch_results)
        current_speed = items_per_batch / avg_batch_time if avg_batch_time > 0 else 0
        overall_speed = processed_items / elapsed if elapsed > 0 else 0
        
        remaining_batches = total_batches - (batch_idx + 1)
        eta_seconds = remaining_batches * avg_batch_time
        
        progress_bar.set_postfix({
            'speed': f'{current_speed:.2f}/s',
            'avg': f'{overall_speed:.2f}/s',
            'batch': f'{current_batch_time:.1f}s',
            'eta': f'{format_time(eta_seconds)}',
            'success': f'{len(batch_results)}/{len(batch)}'
        })
        
        if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(results, batch_idx)
        
        if (batch_idx + 1) % 50 == 0:
            captioner.cleanup_memory()
            gc.collect()
        
        if (batch_idx + 1) % 25 == 0:
            remaining_batches = total_batches - (batch_idx + 1)
            completion_percentage = (batch_idx + 1) / total_batches * 100
            
            print(f"\n{'='*60}")
            print(f"📊 STATISTICHE BATCH {batch_idx + 1}/{total_batches}")
            print(f"{'='*60}")
            print(f"⏱️  Tempo trascorso: {format_time(elapsed)}")
            print(f"🎯 Completamento: {completion_percentage:.1f}%")
            print(f"📈 Velocità corrente: {current_speed:.2f} immagini/sec")
            print(f"📊 Velocità media: {overall_speed:.2f} immagini/sec")
            print(f"⚡ Tempo medio per batch: {avg_batch_time:.1f}s")
            print(f"🏁 Tempo rimanente stimato: {format_time(eta_seconds)}")
            print(f"✅ Immagini processate: {processed_items}/{len(template_data)}")
            print(f"💾 Successi totali: {processed_items}")
            
            completion_time = time.time() + eta_seconds
            completion_datetime = time.strftime("%H:%M:%S del %d/%m/%Y", time.localtime(completion_time))
            print(f"🕐 Completamento stimato: {completion_datetime}")
            print(f"{'='*60}\n")
    
    captioner.cleanup_memory()
    
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    
    if CHECKPOINT_JSON.exists():
        CHECKPOINT_JSON.unlink()
    
    final_speed = len(results) / (time.time() - start_time)
    
    total_time = (time.time() - start_time) / 60

    print(f"\n{'='*60}")
    print(f"🎉 PROCESSO COMPLETATO CON SUCCESSO!")
    print(f"{'='*60}")
    print(f"⏱️  Tempo totale: {format_time(total_time * 60)}")
    print(f"📁 File salvato: {OUTPUT_JSON}")
    print(f"✅ Immagini elaborate: {len(results)}/{len(template_data)}")
    print(f"📊 Tasso di successo: {len(results)/len(template_data)*100:.1f}%")
    print(f"🚀 Velocità media finale: {final_speed:.2f} immagini/sec")
    print(f"💾 Dimensione file output: {OUTPUT_JSON.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"🕐 Completato alle: {time.strftime('%H:%M:%S del %d/%m/%Y')}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
