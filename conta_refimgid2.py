# Script che serve solo per confrontare i risultati di Pavan
# Script che serve per contare se le immagini di cui abbiamo ottenuto descrizioni con DAM sono 32k
# Se non lo sono ci restituisce la lista di ref_image_id che non sono presenti 
# Script che serve per contare se le immagini di cui abbiamo ottenuto descrizioni con DAM sono 32k
# Se non lo sono ci restituisce la lista di ref_image_id che non sono presenti 
import json

def normalize_ref_id(ref_id):
    """Normalizza l'ID rimuovendo gli zeri iniziali e convertendo in intero o stringa"""
    if isinstance(ref_id, (int, float)):
        return str(int(ref_id))
    elif isinstance(ref_id, str):
        # Rimuove gli zeri iniziali e converte
        return str(int(ref_id)) if ref_id.isdigit() else ref_id.lstrip('0')
    else:
        return str(ref_id)

def analyze_file_structure(file_path, normalize_ids=False):
    """Analizza la struttura del file JSON per capire com'è organizzato"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print(f"\n=== ANALISI STRUTTURA DI {file_path} ===")
        
        if isinstance(data, list):
            print(f"Il file contiene una lista di {len(data)} elementi")
            
            # Conta elementi con e senza ref_image_id
            with_ref_id = 0
            without_ref_id = 0
            total_ref_ids = set()
            
            for i, item in enumerate(data[:5]):  # Analizza solo i primi 5 per esempio
                print(f"\nElemento {i}: {type(item)}")
                if isinstance(item, dict):
                    print(f"  Keys: {list(item.keys())}")
                    if "ref_image_id" in item:
                        ref_id = item["ref_image_id"]
                        if normalize_ids:
                            ref_id = normalize_ref_id(ref_id)
                        print(f"  ref_image_id: {ref_id} (originale: {item['ref_image_id']})")
            
            # Analisi completa
            for item in data:
                if isinstance(item, dict):
                    if "ref_image_id" in item:
                        with_ref_id += 1
                        ref_id = item["ref_image_id"]
                        if normalize_ids:
                            ref_id = normalize_ref_id(ref_id)
                        total_ref_ids.add(ref_id)
                    else:
                        without_ref_id += 1
            
            print(f"\nElementi con ref_image_id: {with_ref_id}")
            print(f"Elementi senza ref_image_id: {without_ref_id}")
            print(f"Ref_image_id unici: {len(total_ref_ids)}")
            
            return total_ref_ids, with_ref_id
            
        elif isinstance(data, dict):
            print("Il file contiene un singolo oggetto")
            print(f"Keys: {list(data.keys())}")
            if "ref_image_id" in data:
                ref_id = data["ref_image_id"]
                if normalize_ids:
                    ref_id = normalize_ref_id(ref_id)
                print(f"ref_image_id: {ref_id} (originale: {data['ref_image_id']})")
                return {ref_id}, 1
            else:
                return set(), 0
        else:
            print(f"Tipo di dato inaspettato: {type(data)}")
            return set(), 0
            
    except Exception as e:
        print(f"Errore nell'analisi di {file_path}: {e}")
        return set(), 0

def detailed_analysis():
    file1 = "./data/files/laion_combined_opt_laion_combined_multi.json"
    file2 = "./data/files/laion_combined_info.json"
    
    print("ANALISI DETTAGLIATA DEI FILE JSON")
    print("=" * 50)
    
    # Analizza entrambi i file in dettaglio
    # Per file1 normalizziamo gli ID (rimuoviamo zeri iniziali)
    ref_ids_opt, count_opt = analyze_file_structure(file1, normalize_ids=True)
    # Per file2 non serve normalizzare
    ref_ids_info, count_info = analyze_file_structure(file2, normalize_ids=False)
    
    print(f"\n=== RISULTATI FINALI ===")
    print(f"Elementi con ref_image_id in opt_laion_combined_multi: {count_opt}")
    print(f"Elementi con ref_image_id in info: {count_info}")
    print(f"Ref_image_id unici in opt_laion_combined_multi: {len(ref_ids_opt)}")
    print(f"Ref_image_id unici in info: {len(ref_ids_info)}")
    
    # Trova i mancanti (confronta gli ID normalizzati)
    missing_ids = ref_ids_info - ref_ids_opt
    print(f"\nRef_image_id mancanti in opt_laion_combined_multi: {len(missing_ids)}")
    
    if missing_ids:
        print("Elenco mancanti:")
        for ref_id in sorted(missing_ids):
            print(f"  - {ref_id}")

# === Analisi errori nelle descrizioni: inizio
def find_error_descriptions():
    file_path = "./data/files/laion_combined_dam_multi.json"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        error_ref_ids = set()
        error_entries = []
        
        if isinstance(data, list):
            for item in data:
                if (isinstance(item, dict) and 
                    "ref_image_id" in item and 
                    "multi_caption_dam" in item):
                    
                    ref_id = item["ref_image_id"]
                    multi_caption = item["multi_caption_dam"]
                    
                    # Controlla se multi_caption_dam è una lista di descrizioni
                    if isinstance(multi_caption, list):
                        for description in multi_caption:
                            if isinstance(description, str) and "Error generating description with" in description:
                                error_ref_ids.add(ref_id)
                                error_entries.append({
                                    'ref_image_id': ref_id,
                                    'error_description': description
                                })
                                break  # Basta un errore per segnare questo ref_id
        
        print(f"Numero totale di ref_image_id con errori: {len(error_ref_ids)}")
        print(f"Numero totale di voci con errori: {len(error_entries)}")
        
        if error_ref_ids:
            print("\nRef_image_id con descrizioni di errore:")
            for ref_id in sorted(error_ref_ids):
                print(f"  - {ref_id}")
            
            print(f"\nEsempi di descrizioni con errore (primi 5):")
            for i, entry in enumerate(error_entries[:5]):
                print(f"{i+1}. ref_image_id: {entry['ref_image_id']}")
                print(f"   Descrizione: {entry['error_description']}")
                print()
        
        return error_ref_ids, error_entries
        
    except Exception as e:
        print(f"Errore: {e}")
        return set(), []

# Versione alternativa che conta anche quanti errori per ogni ref_image_id
def detailed_error_analysis():
    file_path = "./data/files/laion_combined_dam_multi.json"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        error_stats = {}
        total_errors = 0
        
        if isinstance(data, list):
            for item in data:
                if (isinstance(item, dict) and 
                    "ref_image_id" in item and 
                    "multi_caption_dam" in item):
                    
                    ref_id = item["ref_image_id"]
                    multi_caption = item["multi_caption_dam"]
                    
                    if isinstance(multi_caption, list):
                        error_count = 0
                        for description in multi_caption:
                            if isinstance(description, str) and "Error generating description with" in description:
                                error_count += 1
                                total_errors += 1
                        
                        if error_count > 0:
                            if ref_id not in error_stats:
                                error_stats[ref_id] = 0
                            error_stats[ref_id] += error_count
        
        print(f"Numero di ref_image_id con errori: {len(error_stats)}")
        print(f"Numero totale di errori trovati: {total_errors}")
        
        if error_stats:
            print("\nRef_image_id con maggiori errori (top 20):")
            sorted_errors = sorted(error_stats.items(), key=lambda x: x[1], reverse=True)
            for ref_id, count in sorted_errors[:20]:
                print(f"  - {ref_id}: {count} errori")
        
        return error_stats, total_errors
        
    except Exception as e:
        print(f"Errore: {e}")
        return {}, 0
# === Analisi errori nelle descrizoni: fine

# Esegui l'analisi dettagliata
detailed_analysis()

# Esegui l'analisi errori nelle descrizioni (commentata per ora)
"""
print("=== ANALISI ERRORI NELLE DESCRIZIONI ===")
error_ref_ids, error_entries = find_error_descriptions()

print("\n" + "="*50)
print("=== ANALISI DETTAGLIATA ERRORI ===")
error_stats, total_errors = detailed_error_analysis()
"""