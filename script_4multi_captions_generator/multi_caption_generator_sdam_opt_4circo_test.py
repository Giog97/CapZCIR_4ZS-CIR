# Script che serve per generare le descrizioni per il dataset CIRCO
"""
1. Carica il file JSON originale test_circo_pavan.json
2. Per ogni elemento: Estrae l'ID dell'immagine reference
3. Carica l'immagine dalla directory CIRCO corretta
4. Genera 15 nuove descrizioni usando CaptionerSDAM_opt
5. Sostituisce il campo multi_caption_opt con multi_caption_dam con le nuove descrizioni
6. Mantiene intatta tutta la struttura originale del JSON
7. Salva il risultato in 'test_circo_dam.json'
"""
import json
from typing import List
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
import gc
from concurrent.futures import ThreadPoolExecutor
import threading

# Configurazione
INPUT_JSON = Path('./data/files/test_circo_pavan.json')
IMAGE_DIR = Path('./data/datasets/CIRCO/COCO2017_unlabeled/unlabeled2017')  # Directory delle immagini CIRCO
OUTPUT_JSON = Path('./data/files/test_circo_dam.json')
CHECKPOINT_JSON = Path('./data/files/checkpoint_test_circo_dam.json')
NUM_CAPTIONS = 15
BATCH_SIZE = 4
CHECKPOINT_INTERVAL = 20
MAX_WORKERS = 4

# Importa il captioner (assicurati che il percorso sia corretto)
import sys
#sys.path.append('./model')  # Modifica se necessario
#from captioner_sdam_opt import CaptionerSDAM_opt
from model.captioner_sdam_opt import CaptionerSDAM_opt

# Inizializza il captioner
print("Inizializzazione del captioner SDAM...")
captioner = CaptionerSDAM_opt()

# Lock per thread safety
results_lock = threading.Lock()

def load_circo_image(ref_image_id: int) -> Image.Image:
    """
    Carica un'immagine CIRCO dal filesystem.
    Le immagini CIRCO hanno path: ./data/datasets/CIRCO/COCO2017_unlabeled/unlabeled2017/{ref_image_id:012d}.jpg
    """
    try:
        image_path = IMAGE_DIR / f"{ref_image_id:012d}.jpg"
        
        if not image_path.exists():
            print(f"❌ Immagine non trovata: {image_path}")
            return None
            
        image = Image.open(image_path).convert('RGB')
        image.load()  # Forza il caricamento
        return image
        
    except Exception as e:
        print(f"❌ Errore caricamento immagine {ref_image_id}: {e}")
        return None

def load_images_batch(ref_image_ids: List[int]) -> List[Image.Image]:
    """Carica un batch di immagini CIRCO in parallelo."""
    def load_single_image(img_id):
        return load_circo_image(img_id)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        images = list(executor.map(load_single_image, ref_image_ids))
    
    # Filtra immagini valide e mantieni traccia degli indici
    valid_images = []
    valid_indices = []
    valid_ids = []
    
    for i, (img, img_id) in enumerate(zip(images, ref_image_ids)):
        if img is not None:
            valid_images.append(img)
            valid_indices.append(i)
            valid_ids.append(img_id)
    
    return valid_images, valid_indices, valid_ids

def process_batch(batch: List[dict]) -> List[dict]:
    """Processa un batch di elementi CIRCO."""
    batch_start_time = time.time()
    
    try:
        # Estrai gli ID delle immagini reference
        ref_image_ids = [item["ref_image_id"] for item in batch]
        
        print(f"🔄 Processando batch con {len(ref_image_ids)} immagini CIRCO...")
        
        # Carica le immagini
        load_start = time.time()
        images, valid_indices, valid_ids = load_images_batch(ref_image_ids)
        load_time = time.time() - load_start
        
        if not images:
            print(f"❌ Nessuna immagine CIRCO valida nel batch")
            return []
        
        print(f"✅ Caricate {len(images)}/{len(ref_image_ids)} immagini in {load_time:.2f}s")
        
        # Genera descrizioni con SDAM
        gen_start = time.time()
        batch_descriptions = captioner.generate_all_descriptions_batch(images)
        gen_time = time.time() - gen_start
        
        print(f"📝 Generazione descrizioni SDAM completata in {gen_time:.2f}s")
        
        # Costruisci i risultati mantenendo la struttura originale
        batch_results = []
        
        for idx, (original_idx, image_id) in enumerate(zip(valid_indices, valid_ids)):
            original_item = batch[original_idx]
            descriptions = batch_descriptions[idx]
            
            if descriptions and len(descriptions) >= NUM_CAPTIONS:
                # Crea una copia dell'item originale sostituendo le descrizioni
                new_item = original_item.copy()
                new_item["multi_caption_dam"] = descriptions[:NUM_CAPTIONS]
                batch_results.append(new_item)
            else:
                desc_count = len(descriptions) if descriptions else 0
                print(f"⚠️  Solo {desc_count} descrizioni per {image_id}")
                # Mantieni l'item originale se non abbiamo descrizioni sufficienti
                batch_results.append(original_item)
        
        # Gestisci gli item con immagini non valide (mantieni originali)
        for i, item in enumerate(batch):
            if i not in valid_indices:
                print(f"⚠️  Mantenuto item originale per {item['ref_image_id']} (immagine non valida)")
                batch_results.append(item)
        
        batch_total_time = time.time() - batch_start_time
        success_rate = len(valid_indices) / len(batch) * 100
        
        print(f"✨ Batch completato: {len(valid_indices)}/{len(batch)} successi ({success_rate:.1f}%) in {batch_total_time:.2f}s")
        
        return batch_results
        
    except Exception as e:
        print(f"💥 Errore nel batch processing: {e}")
        import traceback
        traceback.print_exc()
        # In caso di errore, ritorna gli item originali
        return batch

def save_checkpoint(results: List[dict], batch_idx: int):
    """Salva un checkpoint dei risultati."""
    checkpoint_data = {
        "batch_processed": batch_idx,
        "results": results,
        "timestamp": time.time()
    }
    with open(CHECKPOINT_JSON, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"💾 Checkpoint salvato dopo batch {batch_idx}")

def load_checkpoint() -> tuple:
    """Carica un checkpoint se esistente."""
    if CHECKPOINT_JSON.exists():
        try:
            with open(CHECKPOINT_JSON, 'r') as f:
                checkpoint = json.load(f)
            return checkpoint["batch_processed"], checkpoint["results"]
        except Exception as e:
            print(f"❌ Errore caricamento checkpoint: {e}")
    return 0, []

def format_time(seconds):
    """Formatta il tempo in ore:minuti:secondi."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    else:
        return f"{minutes:02d}m {secs:02d}s"

def main():
    # Carica il file JSON originale
    print(f"📖 Caricamento file {INPUT_JSON}...")
    with open(INPUT_JSON, 'r') as f:
        original_data = json.load(f)
    
    # Carica checkpoint se disponibile
    start_batch, results = load_checkpoint()
    
    total_batches = (len(original_data) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()
    
    print(f"\n🚀 AVVIO ELABORAZIONE CIRCO")
    print(f"{'='*60}")
    print(f"📁 File input: {INPUT_JSON}")
    print(f"📊 Elementi totali: {len(original_data)}")
    print(f"📦 Batch totali: {total_batches} (da {BATCH_SIZE} elementi)")
    print(f"🎯 Descrizioni per immagine: {NUM_CAPTIONS}")
    print(f"💾 Checkpoint ogni {CHECKPOINT_INTERVAL} batch")
    print(f"⚙️  Device: {captioner.device}")
    
    if start_batch > 0:
        print(f"🔄 Ripresa dal batch {start_batch + 1}/{total_batches}")
        print(f"✅ Già processati: {len(results)} elementi")
    
    print(f"{'='*60}")
    
    # Processa i batch
    progress_bar = tqdm(
        range(start_batch, total_batches), 
        desc="Batch processati",
        initial=start_batch,
        total=total_batches,
        unit="batch"
    )
    
    for batch_idx in progress_bar:
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(original_data))
        batch = original_data[start_idx:end_idx]
        
        # Processa il batch
        batch_results = process_batch(batch)
        
        # Aggiungi i risultati
        with results_lock:
            results.extend(batch_results)
        
        # Aggiorna la progress bar
        processed = len(results)
        elapsed = time.time() - start_time
        progress_percent = processed / len(original_data) * 100
        
        progress_bar.set_postfix({
            'completamento': f'{progress_percent:.1f}%',
            'elementi': f'{processed}/{len(original_data)}'
        })
        
        # Salva checkpoint periodicamente
        if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(results, batch_idx + 1)
        
        # Pulizia memoria
        if (batch_idx + 1) % 10 == 0:
            captioner.cleanup_memory()
            gc.collect()
    
    # Salva il file finale
    print(f"\n💾 Salvataggio file finale {OUTPUT_JSON}...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Rimuovi checkpoint
    if CHECKPOINT_JSON.exists():
        CHECKPOINT_JSON.unlink()
    
    # Statistiche finali
    total_time = time.time() - start_time
    print(f"\n🎉 PROCESSO COMPLETATO!")
    print(f"{'='*60}")
    print(f"⏱️  Tempo totale: {format_time(total_time)}")
    print(f"📁 File salvato: {OUTPUT_JSON}")
    print(f"✅ Elementi processati: {len(results)}")
    print(f"🚀 Velocità media: {len(results)/total_time:.2f} elementi/sec")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()