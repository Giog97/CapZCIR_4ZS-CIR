# Valutazione Descrizioni SDAM generiche (sia checkpoint che file finali) con BLEU
import json
import evaluate
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import warnings
from transformers.utils import logging

# Configurazione paths
# Le seguenti due directory servono per selezionare il json da analizzare e indicare il file dove salvare
# quindi, se si vuole analizzare un altro risultato di un captioner basta cambiare queste directory.
#file_path = Path('./data/files/laion_combined_dam_multi.json')   # finale o checkpoint
file_path = Path('./data/files/laion_combined_dam_multi_fixed.json')   # finale o checkpoint
original_captions_path = Path('./data/files/exist_caption_first_million.json') # descrizioni originali
#output_json_path = Path('./data/files/bleu_scores_sdam_auto.json') # risultati metrica BLEU
output_json_path = Path('./data/files/bleu_scores_sdam_auto_fixed.json') # risultati metrica BLEU

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")
pd.set_option('display.float_format', '{:.6f}'.format)

# Carica descrizioni originali
print("Caricamento descrizioni originali...")
with open(original_captions_path, 'r') as f:
    original_captions_data = json.load(f)

original_captions_dict = {str(k): v for k, v in original_captions_data.items()}

# Carica dati SDAM (può essere checkpoint o finale)
print("Caricamento descrizioni generate...")
with open(file_path, 'r') as f:
    data = json.load(f)

# Normalizza formato: se è checkpoint ha "results", se è finale è lista
if isinstance(data, dict) and "results" in data:
    entries = data["results"]
elif isinstance(data, list):
    entries = data
else:
    raise ValueError("Formato JSON non riconosciuto: atteso dict con 'results' o lista di entries")

# Initialize BLEU
bleu = evaluate.load("bleu")

# Track execution time
start_time = time.time()

all_preds, all_refs = [], []
results = []
missing_captions, total_comparisons = 0, 0

for entry in tqdm(entries, desc="Evaluating BLEU scores"):
    ref_image_id = entry['ref_image_id']
    hypotheses = entry['multi_caption_dam']

    # Cerca descrizione originale
    original_caption = original_captions_dict.get(str(ref_image_id))

    if original_caption is None:
        missing_captions += 1
        continue

    for caption in hypotheses:
        # Sentence-level BLEU
        score = bleu.compute(
            predictions=[caption],
            references=[[original_caption]],
            smooth=True
        )

        results.append({
            'ref_image_id': ref_image_id,
            'original_caption': original_caption,
            'generated_caption': caption,
            'bleu_score': score['bleu']
        })
        total_comparisons += 1

        # Corpus-level
        all_preds.append(caption)
        all_refs.append([original_caption])

# Corpus BLEU
corpus_bleu = bleu.compute(
    predictions=all_preds,
    references=all_refs,
    smooth=True
)

execution_time = time.time() - start_time
df = pd.DataFrame(results)

# Salva risultati
with open(output_json_path, 'w') as f:
    json.dump(results, f, indent=4)

# Statistiche
print("\n--- Risultati ---")
print(f"Confronti effettuati: {total_comparisons}")
print(f"Descrizioni originali non trovate: {missing_captions}")
print(f"Percentuale successo: {(len(entries) - missing_captions) / len(entries) * 100:.2f}%")

if not df.empty:
    print("\n--- Sentence-level BLEU ---")
    print(f"BLEU Medio: {df['bleu_score'].mean():.6f}")
    print(f"BLEU Massimo: {df['bleu_score'].max():.6f}")
    print(f"BLEU Minimo: {df['bleu_score'].min():.6f}")

print("\n--- Corpus-level BLEU ---")
print(f"BLEU Corpus: {corpus_bleu['bleu']:.6f}")

print(f"\nTempo esecuzione: {execution_time:.2f} secondi")
print(f"Risultati salvati in: {output_json_path}")

# Statistiche aggiuntive
stats = {
    'total_entries': len(entries),
    'successful_comparisons': total_comparisons,
    'missing_original_captions': missing_captions,
    'mean_sentence_bleu': df['bleu_score'].mean() if not df.empty else 0,
    'corpus_bleu': corpus_bleu['bleu'],
    'execution_time_seconds': execution_time
}

with open(output_json_path.parent / 'bleu_stats_auto.json', 'w') as f:
    json.dump(stats, f, indent=4)


