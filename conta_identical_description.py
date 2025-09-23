import json
import re
from collections import defaultdict, Counter

def normalize_text(text):
    """Normalizza il testo: minuscolo e rimuove punteggiatura"""
    # Converti in minuscolo
    text = text.lower()
    # Rimuovi punteggiatura (mantieni spazi e lettere)
    text = re.sub(r'[^\w\s]', '', text)
    # Rimuovi spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_duplicate_descriptions(json_file_path, use_normalization=False, detailed_output=False):
    """
    Analizza le descrizioni duplicate nel file JSON
    
    Args:
        json_file_path: percorso del file JSON da analizzare
        use_normalization: se True, usa normalizzazione case-insensitive e senza punteggiatura
        detailed_output: se True, include informazioni dettagliate sui duplicati
    """
    # Carica il file JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Dizionari per tracciare i risultati
    images_with_duplicates = set()
    duplicate_counts = Counter()
    images_by_duplicate_count = defaultdict(list)
    all_duplicates_info = [] if detailed_output else None
    
    # Analizza ogni elemento
    for item in data:
        ref_image_id = item["ref_image_id"]
        descriptions = item["multi_caption_dam"]
        
        if use_normalization:
            # Normalizza tutte le descrizioni
            processed_descriptions = [normalize_text(desc) for desc in descriptions]
        else:
            # Usa descrizioni originali
            processed_descriptions = descriptions
        
        # Conta le occorrenze di ogni descrizione
        description_counts = Counter(processed_descriptions)
        
        # Trova descrizioni duplicate (che compaiono più di una volta)
        duplicates = {desc: count for desc, count in description_counts.items() if count > 1}
        
        if duplicates:
            images_with_duplicates.add(ref_image_id)
            
            # Calcola il numero massimo di duplicati per la stessa descrizione
            max_duplicate_count = max(duplicates.values()) if duplicates else 0
            
            # Aggiorna i contatori
            duplicate_counts[max_duplicate_count] += 1
            images_by_duplicate_count[max_duplicate_count].append(ref_image_id)
            
            # Salva informazioni dettagliate se richiesto
            if detailed_output:
                duplicate_info = {
                    "ref_image_id": ref_image_id,
                    "duplicate_groups": []
                }
                
                # Trova i gruppi di descrizioni duplicate
                for proc_desc, count in duplicates.items():
                    if count > 1:
                        # Trova tutte le descrizioni originali che corrispondono
                        original_duplicates = []
                        for i, orig_desc in enumerate(descriptions):
                            if use_normalization:
                                # Confronta con versione normalizzata
                                if normalize_text(orig_desc) == proc_desc:
                                    original_duplicates.append(orig_desc)
                            else:
                                # Confronto esatto
                                if orig_desc == proc_desc:
                                    original_duplicates.append(orig_desc)
                        
                        duplicate_info["duplicate_groups"].append({
                            "processed_text": proc_desc if not use_normalization else proc_desc,
                            "count": count,
                            "original_descriptions": original_duplicates
                        })
                
                all_duplicates_info.append(duplicate_info)
    
    # Calcola le statistiche
    total_images_with_duplicates = len(images_with_duplicates)
    total_images = len(data)
    
    # Prepara i risultati
    results = {
        "total_images": total_images,
        "total_images_with_duplicates": total_images_with_duplicates,
        "analysis_type": "normalized" if use_normalization else "exact_match",
        "duplicate_breakdown": {f"images_with_{count}_identical_descriptions": cnt 
                               for count, cnt in duplicate_counts.items()},
        "images_with_duplicates": list(images_with_duplicates),
        "detailed_breakdown": {f"{count}_identical_descriptions": images 
                              for count, images in images_by_duplicate_count.items()}
    }
    
    # Aggiungi informazioni dettagliate se richiesto
    if detailed_output:
        results["duplicates_detailed_info"] = all_duplicates_info
    
    return results

def save_results_to_json(results, output_file_path):
    """Salva i risultati in un file JSON"""
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def print_results(results):
    """Stampa i risultati in formato leggibile"""
    analysis_type = results["analysis_type"]
    print(f"Numero totale di immagini: {results['total_images']}")
    print(f"Numero di immagini con almeno una descrizione duplicata ({analysis_type}): {results['total_images_with_duplicates']}")
    print("\nDettaglio duplicati:")
    for count_key, count_value in results['duplicate_breakdown'].items():
        print(f"{count_key}: {count_value}")

# === CONFIGURAZIONE ===
# Scegli quale analisi eseguire cambiando questi parametri:
USE_NORMALIZATION = True  # True per case-insensitive senza punteggiatura, False per confronto esatto
DETAILED_OUTPUT = True    # True per output dettagliato, False per output base

json_file_path = "./data/files/laion_combined_dam_multi_fixed.json"
#output_file_path = "./data/files/duplicate_analysis_results.json" # USE_NORMALIZATION = False
output_file_path = "./data/files/duplicate_analysis_results_norm.json" # USE_NORMALIZATION = True

# === ESECUZIONE ===
results = analyze_duplicate_descriptions(
    json_file_path, 
    use_normalization=USE_NORMALIZATION,
    detailed_output=DETAILED_OUTPUT
)

# Stampa i risultati
print_results(results)

# Salva i risultati
save_results_to_json(results, output_file_path)
print(f"\nRisultati salvati in: {output_file_path}")

