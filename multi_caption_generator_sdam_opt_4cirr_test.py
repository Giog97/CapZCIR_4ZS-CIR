# Script che serve per generare le descrizioni per il dataset CIRR e poter così fare test con CIRR
# NB implementato per cap.rc2.test1.json 
"""
1. Carica il file JSON originale cap.rc2.test1_fixed.json (partenza da file di PAVAN)
--> se si vuole partite dal file originale di CIRR bisogna modificare questo script (file originale cap.rc2.test1.json)
2. Per ogni elemento: Estrae il nome dell'immagine reference
3. Carica l'immagine dalla directory CIRR corretta
4. Genera 15 nuove descrizioni usando CaptionerSDAM_opt
5. Sostituisce il campo multi_caption_dam con le nuove descrizioni (questo campo è presente solo nel file di PAVAN modificato)
--> se si parte dal file originale di CIRR bisogna aggiungere questo nuovo campo con chiave multi_caption_dam e seguito dalle descrizioni
6. Mantiene intatta tutta la struttura originale del JSON
7. Salva il risultato in 'cap.rc2.test1_dam.json'
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
INPUT_JSON = Path('./data/files/cap.rc2.test1_fixed.json')
IMAGE_DIR = Path('./data/datasets/CIRR')  # Directory delle immagini CIRR
OUTPUT_JSON = Path('./data/files/cap.rc2.test1_dam.json')
CHECKPOINT_JSON = Path('./data/files/checkpoint_cirr_test1_dam.json')
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

def load_cirr_image(image_name: str) -> Image.Image:
    """
    Carica un'immagine CIRR dal filesystem.
    Le immagini CIRR hanno path: ./data/datasets/CIRR/{split}/{image_name}.png
    """
    try:
        # Estrai lo split dal nome dell'immagine (es: "dev-244-0-img0" -> "dev")
        split = image_name.split('-')[0]
        image_path = IMAGE_DIR / split / f"{image_name}.png" # per val --> split=dev, per test --> split=test1
        
        if not image_path.exists():
            print(f"❌ Immagine non trovata: {image_path}")
            return None
            
        image = Image.open(image_path).convert('RGB')
        image.load()  # Forza il caricamento
        return image
        
    except Exception as e:
        print(f"❌ Errore caricamento immagine {image_name}: {e}")
        return None

def load_images_batch(image_names: List[str]) -> List[Image.Image]:
    """Carica un batch di immagini CIRR in parallelo."""
    def load_single_image(img_name):
        return load_cirr_image(img_name)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        images = list(executor.map(load_single_image, image_names))
    
    # Filtra immagini valide e mantieni traccia degli indici
    valid_images = []
    valid_indices = []
    valid_names = []
    
    for i, (img, name) in enumerate(zip(images, image_names)):
        if img is not None:
            valid_images.append(img)
            valid_indices.append(i)
            valid_names.append(name)
    
    return valid_images, valid_indices, valid_names

def process_batch(batch: List[dict]) -> List[dict]:
    """Processa un batch di elementi CIRR."""
    batch_start_time = time.time()
    
    try:
        # Estrai i nomi delle immagini reference
        image_names = [item["reference"] for item in batch]
        
        print(f"🔄 Processando batch con {len(image_names)} immagini CIRR...")
        
        # Carica le immagini
        load_start = time.time()
        images, valid_indices, valid_names = load_images_batch(image_names)
        load_time = time.time() - load_start
        
        if not images:
            print(f"❌ Nessuna immagine CIRR valida nel batch")
            return []
        
        print(f"✅ Caricate {len(images)}/{len(image_names)} immagini in {load_time:.2f}s")
        
        # Genera descrizioni con SDAM
        gen_start = time.time()
        batch_descriptions = captioner.generate_all_descriptions_batch(images)
        gen_time = time.time() - gen_start
        
        print(f"📝 Generazione descrizioni SDAM completata in {gen_time:.2f}s")
        
        # Costruisci i risultati mantenendo la struttura originale
        batch_results = []
        
        for idx, (original_idx, image_name) in enumerate(zip(valid_indices, valid_names)):
            original_item = batch[original_idx]
            descriptions = batch_descriptions[idx]
            
            if descriptions and len(descriptions) >= NUM_CAPTIONS:
                # Crea una copia dell'item originale sostituendo solo le descrizioni
                new_item = original_item.copy()
                new_item["multi_caption_dam"] = descriptions[:NUM_CAPTIONS]
                batch_results.append(new_item)
            else:
                desc_count = len(descriptions) if descriptions else 0
                print(f"⚠️  Solo {desc_count} descrizioni per {image_name}")
                # Mantieni l'item originale se non abbiamo descrizioni sufficienti
                batch_results.append(original_item)
        
        # Gestisci gli item con immagini non valide (mantieni originali)
        for i, item in enumerate(batch):
            if i not in valid_indices:
                print(f"⚠️  Mantenuto item originale per {item['reference']} (immagine non valida)")
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
    
    print(f"\n🚀 AVVIO ELABORAZIONE CIRR")
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