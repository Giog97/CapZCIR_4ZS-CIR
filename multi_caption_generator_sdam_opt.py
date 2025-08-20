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

# Configurazione ottimizzata
INPUT_JSON = Path('./data/files/laion_combined_info.json')
IMAGE_DIR = Path('./data/datasets/laion_cir_combined')
OUTPUT_JSON = Path('./data/files/laion_combined_dam_multi.json')
CHECKPOINT_JSON = Path('./data/files/checkpoint_dam_multi.json')
NUM_CAPTIONS = 15
BATCH_SIZE = 4  # Ridotto a 4 per debug iniziale - aumenta gradualmente
CHECKPOINT_INTERVAL = 50  # Salva più frequentemente per debug
MAX_WORKERS = 4  # Ridotto a 2 per evitare sovraccarico

# Inizializza il captioner una sola volta
print("Inizializzazione del captioner...")
captioner = CaptionerSDAM_opt()

# Lock per thread safety
results_lock = threading.Lock()

def load_images_parallel(image_ids: List[str]) -> List[Image.Image]:
    """Carica le immagini in parallelo usando ThreadPoolExecutor."""
    def load_single_image(image_id: str) -> Image.Image:
        try:
            image_path = IMAGE_DIR / f"{image_id.zfill(7)}.png"
            image = Image.open(image_path).convert('RGB')
            # Verifica che l'immagine sia valida
            image.load()  # Forza il caricamento per catturare errori di corruzione
            return image
        except Exception as e:
            print(f"Errore caricamento immagine {image_id}: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        images = list(executor.map(load_single_image, image_ids))
    
    # Filtra le immagini non caricate e restituisci anche gli ID validi
    valid_pairs = [(img, img_id) for img, img_id in zip(images, image_ids) if img is not None]
    if not valid_pairs:
        return [], []
    
    valid_images, valid_ids = zip(*valid_pairs)
    return list(valid_images), list(valid_ids)

def format_time(seconds):
    """Converte secondi in formato leggibile (ore:minuti:secondi)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    else:
        return f"{minutes:02d}m {secs:02d}s"

def process_batch(batch: List[dict]) -> List[dict]:
    """Processa un batch di elementi usando il batch processing ottimizzato."""
    batch_start_time = time.time()
    try:
        # Estrai gli ID delle immagini
        image_ids = [item["ref_image_id"] for item in batch]
        
        print(f"🔄 Processando batch con {len(image_ids)} immagini...")
        
        # Carica tutte le immagini del batch in parallelo
        load_start = time.time()
        images, valid_ids = load_images_parallel(image_ids)
        load_time = time.time() - load_start
        
        if not images:
            print(f"❌ Nessuna immagine valida nel batch")
            return []
        
        print(f"✅ Caricate {len(images)}/{len(image_ids)} immagini in {load_time:.2f}s")
        
        # Genera tutte le descrizioni del batch contemporaneamente
        gen_start = time.time()
        batch_descriptions = captioner.generate_all_descriptions_batch(images)
        gen_time = time.time() - gen_start
        
        print(f"📝 Generazione descrizioni completata in {gen_time:.2f}s")
        
        # Costruisci i risultati
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

def save_checkpoint(results: List[dict], batch_idx: int):
    """Salva un checkpoint dei risultati."""
    checkpoint_data = {
        "batch_processed": batch_idx,
        "results": results,
        "timestamp": time.time()
    }
    with open(CHECKPOINT_JSON, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"Checkpoint salvato dopo batch {batch_idx}")

def load_checkpoint() -> tuple:
    """Carica un checkpoint se esistente."""
    if CHECKPOINT_JSON.exists():
        try:
            with open(CHECKPOINT_JSON, 'r') as f:
                checkpoint = json.load(f)
            return checkpoint["batch_processed"], checkpoint["results"]
        except Exception as e:
            print(f"Errore caricamento checkpoint: {e}")
    return 0, []

def main():
    # Carica il template JSON
    with open(INPUT_JSON, 'r') as f:
        template_data = json.load(f)
    
    # Carica checkpoint se disponibile
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
    
    # Processa per batch con progress bar ottimizzata
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
        
        # Processa il batch
        batch_start = time.time()
        batch_results = process_batch(batch)
        batch_time = time.time() - batch_start
        
        # Thread-safe append
        with results_lock:
            results.extend(batch_results)
        
        # Aggiorna statistiche con calcoli più precisi
        processed_items = len(results)
        elapsed = time.time() - start_time
        current_batch_time = time.time() - batch_start
        
        # Calcola velocità su finestra mobile (ultimi 10 batch)
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
        
        # Calcola ETA basato su velocità recente
        remaining_batches = total_batches - (batch_idx + 1)
        eta_seconds = remaining_batches * avg_batch_time
        
        # Aggiorna progress bar con informazioni dettagliate
        progress_bar.set_postfix({
            'speed': f'{current_speed:.2f}/s',
            'avg': f'{overall_speed:.2f}/s',
            'batch': f'{current_batch_time:.1f}s',
            'eta': f'{format_time(eta_seconds)}',
            'success': f'{len(batch_results)}/{len(batch)}'
        })
        
        # Salva checkpoint periodicamente
        if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(results, batch_idx)
        
        # Pulizia memoria ogni 50 batch
        if (batch_idx + 1) % 50 == 0:
            captioner.cleanup_memory()
            gc.collect()
        
        # Stampa statistiche dettagliate ogni 25 batch
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
            
            # Calcola ora di completamento stimata
            completion_time = time.time() + eta_seconds
            completion_datetime = time.strftime("%H:%M:%S del %d/%m/%Y", time.localtime(completion_time))
            print(f"🕐 Completamento stimato: {completion_datetime}")
            print(f"{'='*60}\n")
    
    # Pulizia finale
    captioner.cleanup_memory()
    
    # Salva i risultati finali
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Rimuovi checkpoint
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