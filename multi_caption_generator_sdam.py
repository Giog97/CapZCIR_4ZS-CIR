import warnings
# Sopprime specificamente solo questo warning
warnings.filterwarnings("ignore", message="`torch.utils._pytree._register_pytree_node` is deprecated")
import json
import torch
from pathlib import Path
from typing import List
from PIL import Image
from tqdm import tqdm
from model.captioner_sdam import CaptionerSDAM
import time

# Per usare seconda GPU usare il seguente comando: CUDA_VISIBLE_DEVICES=1 python multi_caption_generator_sdam.py

# Configurazione
INPUT_JSON = Path('./data/files/laion_combined_info.json')
IMAGE_DIR = Path('./data/datasets/laion_cir_combined')
OUTPUT_JSON = Path('./data/files/laion_combined_dam_multi.json')
NUM_CAPTIONS = 15
BATCH_SIZE = 8  # Numero di immagini da processare per batch impostarlo piccolo

# Inizializza il captioner
captioner = CaptionerSDAM()

def process_batch(batch: List[dict]) -> List[dict]:
    """Processa un batch di elementi e genera le caption"""
    batch_results = []
    for item in batch:
        try:
            ref_image_id = item["ref_image_id"]
            captions = process_image(ref_image_id)
            
            if not captions:
                print(f"\nAttenzione: nessuna caption generata per {ref_image_id}")
                continue
            
            batch_results.append({
                "ref_image_id": ref_image_id,
                "relative_cap": item["relative_cap"],
                "tgt_image_id": item["tgt_image_id"],
                "multi_caption_dam": captions
            })
        except Exception as e:
            print(f"\nErrore processando l'item {item}: {e}")
    return batch_results

def process_image(image_id: str) -> List[str]:
    """Carica un'immagine e genera le caption"""
    try:
        image_path = IMAGE_DIR / f"{image_id.zfill(7)}.png"
        image = Image.open(image_path).convert('RGB')
        descriptions = captioner.generate_all_descriptions(image)
        return descriptions[:NUM_CAPTIONS]
    except Exception as e:
        print(f"\nErrore processando l'immagine {image_id}: {e}")
        return []

def main():
    # Carica il template JSON
    with open(INPUT_JSON, 'r') as f:
        template_data = json.load(f)
    
    results = []
    total_items = len(template_data)
    processed_items = 0
    start_time = time.time()
    
    print(f"\nInizio elaborazione di {total_items} immagini in batch da {BATCH_SIZE}...")
    
    # Processa per batch
    for i in tqdm(range(0, total_items, BATCH_SIZE), desc="Progresso batch"):
        batch = template_data[i:i + BATCH_SIZE]
        batch_results = process_batch(batch)
        results.extend(batch_results)
        processed_items += len(batch_results)
        
        # Stampa informazioni periodiche
        if (i // BATCH_SIZE) % 10 == 0:  # Ogni 10 batch
            elapsed = time.time() - start_time
            items_per_sec = processed_items / elapsed if elapsed > 0 else 0
            remaining = (total_items - processed_items) / items_per_sec if items_per_sec > 0 else 0
            
            print(f"\nStato avanzamento:")
            print(f"- Immagini processate: {processed_items}/{total_items} ({processed_items/total_items:.1%})")
            print(f"- Velocità: {items_per_sec:.2f} immagini/sec")
            print(f"- Tempo trascorso: {elapsed/60:.1f} minuti")
            print(f"- Tempo stimato rimanente: {remaining/60:.1f} minuti")
    
    # Salva i risultati
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=4)
    
    total_time = (time.time() - start_time) / 60
    print(f"\nProcesso completato in {total_time:.1f} minuti")
    print(f"Risultati salvati in {OUTPUT_JSON}")
    print(f"Immagini processate con successo: {len(results)}/{total_items}")

if __name__ == "__main__":
    main()