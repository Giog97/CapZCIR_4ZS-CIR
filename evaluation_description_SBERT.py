# Valutazione Descrizioni SDAM con SBERT [Sentence-BERT + Cosine Similarity]
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import warnings
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Configurazione paths
file_path = Path('./data/files/laion_combined_dam_multi_fixed.json')
original_captions_path = Path('./data/files/exist_caption_first_million.json')
output_json_path = Path('./data/files/sbert_scores_sdam_auto_fixed.json')

# Suppress warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', '{:.6f}'.format)

# Carica descrizioni originali
print("Caricamento descrizioni originali...")
with open(original_captions_path, 'r') as f:
    original_captions_data = json.load(f)

original_captions_dict = {str(k): v for k, v in original_captions_data.items()}

# Carica dati SDAM
print("Caricamento descrizioni generate...")
with open(file_path, 'r') as f:
    data = json.load(f)

# Normalizza formato
if isinstance(data, dict) and "results" in data:
    entries = data["results"]
elif isinstance(data, list):
    entries = data
else:
    raise ValueError("Formato JSON non riconosciuto")

# Inizializza modello e tokenizer
print("Caricamento modello all-MiniLM-L6-v2...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Sposta il modello su GPU se disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Funzione per ottenere embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    
    return mean_pooled

# Funzione per calcolare cosine similarity
def cosine_similarity(emb1, emb2):
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    return torch.mm(emb1, emb2.transpose(0, 1)).item()

# Track execution time
start_time = time.time()

results = []
missing_captions, total_comparisons = 0, 0

# Lista per cosine similarity a livello di corpus
all_original_embeddings = []
all_generated_embeddings = []

for entry in tqdm(entries, desc="Evaluating SBERT scores"):
    ref_image_id = entry['ref_image_id']
    hypotheses = entry['multi_caption_dam']

    # Cerca descrizione originale
    original_caption = original_captions_dict.get(str(ref_image_id))

    if original_caption is None:
        missing_captions += 1
        continue

    # Codifica la descrizione originale una volta
    original_embedding = get_embedding(original_caption)
    all_original_embeddings.append(original_embedding.cpu())

    for caption in hypotheses:
        # Codifica la descrizione generata
        generated_embedding = get_embedding(caption)
        all_generated_embeddings.append(generated_embedding.cpu())
        
        # Calcola cosine similarity
        cosine_score = cosine_similarity(original_embedding, generated_embedding)
        
        results.append({
            'ref_image_id': ref_image_id,
            'original_caption': original_caption,
            'generated_caption': caption,
            'sbert_score': cosine_score
        })
        total_comparisons += 1

# Calcola cosine similarity media a livello di corpus
if all_original_embeddings and all_generated_embeddings:
    original_embeddings_tensor = torch.cat(all_original_embeddings)
    generated_embeddings_tensor = torch.cat(all_generated_embeddings)
    
    # Normalizza gli embeddings
    original_embeddings_tensor = F.normalize(original_embeddings_tensor, p=2, dim=1)
    generated_embeddings_tensor = F.normalize(generated_embeddings_tensor, p=2, dim=1)
    
    # Calcola cosine similarity per tutte le coppie
    corpus_cosine_scores = torch.mm(original_embeddings_tensor, generated_embeddings_tensor.t())
    # Prendi la diagonale (coppie corrispondenti)
    corpus_cosine_mean = corpus_cosine_scores.diag().mean().item()
else:
    corpus_cosine_mean = 0

execution_time = time.time() - start_time
df = pd.DataFrame(results)

# Salva risultati
with open(output_json_path, 'w') as f:
    json.dump(results, f, indent=4)

# Statistiche
print("\n--- Risultati SBERT + Cosine Similarity ---")
print(f"Confronti effettuati: {total_comparisons}")
print(f"Descrizioni originali non trovate: {missing_captions}")
print(f"Percentuale successo: {(len(entries) - missing_captions) / len(entries) * 100:.2f}%")

if not df.empty:
    print("\n--- Sentence-level Cosine Similarity ---")
    print(f"Cosine Similarity Media: {df['sbert_score'].mean():.6f}")
    print(f"Cosine Similarity Massima: {df['sbert_score'].max():.6f}")
    print(f"Cosine Similarity Minima: {df['sbert_score'].min():.6f}")
    print(f"Deviazione Standard: {df['sbert_score'].std():.6f}")

print("\n--- Corpus-level Cosine Similarity ---")
print(f"Cosine Similarity Corpus: {corpus_cosine_mean:.6f}")

print(f"\nTempo esecuzione: {execution_time:.2f} secondi")
print(f"Risultati salvati in: {output_json_path}")

# Statistiche aggiuntive
stats = {
    'total_entries': len(entries),
    'successful_comparisons': total_comparisons,
    'missing_original_captions': missing_captions,
    'mean_sentence_cosine': df['sbert_score'].mean() if not df.empty else 0,
    'max_sentence_cosine': df['sbert_score'].max() if not df.empty else 0,
    'min_sentence_cosine': df['sbert_score'].min() if not df.empty else 0,
    'std_sentence_cosine': df['sbert_score'].std() if not df.empty else 0,
    'corpus_cosine': corpus_cosine_mean,
    'execution_time_seconds': execution_time
}

with open(output_json_path.parent / 'sbert_stats_auto.json', 'w') as f:
    json.dump(stats, f, indent=4)

with open(output_json_path.parent / 'sbert_stats_auto.json', 'w') as f:
    json.dump(stats, f, indent=4)