# Script usato per fixare le descrizioni ottenute con SDAM:
# inserito le 15 descrizioni ottenute da Pavan con BLIP per le 40 immagini
# che hanno dato errore durante l’elaborazione con SDAM
import json

def normalize_ref_id(ref_id):
    """Normalizza l'ID rimuovendo gli zeri iniziali"""
    if isinstance(ref_id, (int, float)):
        return str(int(ref_id))
    elif isinstance(ref_id, str):
        return str(int(ref_id)) if ref_id.isdigit() else ref_id.lstrip('0')
    else:
        return str(ref_id)

def load_json_file(file_path):
    """Carica un file JSON"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Errore nel caricamento di {file_path}: {e}")
        return None

def create_opt_mapping(opt_data):
    """Crea una mappa ref_image_id -> multi_caption_opt dal file opt"""
    opt_mapping = {}
    if isinstance(opt_data, list):
        for item in opt_data:
            if isinstance(item, dict) and "ref_image_id" in item and "multi_caption_opt" in item:
                ref_id = normalize_ref_id(item["ref_image_id"])
                opt_mapping[ref_id] = item["multi_caption_opt"]
    return opt_mapping

def fix_dam_errors_and_add_missing():
    # Carica tutti i file
    dam_data = load_json_file("./data/files/laion_combined_dam_multi.json")
    opt_data = load_json_file("./data/files/laion_combined_opt_laion_combined_multi.json")
    info_data = load_json_file("./data/files/laion_combined_info.json")
    
    if dam_data is None or opt_data is None:
        return
    
    # Crea mappatura delle descrizioni OPT
    opt_mapping = create_opt_mapping(opt_data)
    print(f"Trovate {len(opt_mapping)} descrizioni OPT nel file")
    
    # Lista di ref_image_id da processare
    error_ref_ids = [
        '114023', '165405', '16554', '217690', '2424', '268129', '390135', 
        '41850', '42650', '43779', '537150', '647006', '71953', '72523', 
        '73443', '81831', '82860', '844066', '881829'
    ]
    
    missing_ref_ids = [
        '105668', '114255', '115975', '115981', '116321', '367924', '517184', 
        '6604', '67627', '72392', '80113', '81123', '827365', '860036', 
        '89194', '90210', '93529', '95895', '95923', '95944', '98180'
    ]
    
    # 1. Correggi gli errori nel file DAM
    fixed_count = 0
    if isinstance(dam_data, list):
        for item in dam_data:
            if isinstance(item, dict) and "ref_image_id" in item:
                ref_id = normalize_ref_id(item["ref_image_id"])
                if ref_id in error_ref_ids and ref_id in opt_mapping:
                    # Sostituisce multi_caption_dam con le descrizioni OPT
                    item["multi_caption_dam"] = opt_mapping[ref_id]
                    fixed_count += 1
                    print(f"Corretto ref_image_id {ref_id}")
    
    print(f"Corretti {fixed_count} elementi con errori nelle descrizioni")
    
    # 2. Aggiungi gli elementi mancanti
    added_count = 0
    if isinstance(dam_data, list):
        for ref_id in missing_ref_ids:
            if ref_id in opt_mapping:
                # Crea un nuovo elemento basato sulla struttura del file INFO
                new_item = None
                
                # Cerca l'elemento corrispondente nel file info
                if isinstance(info_data, list):
                    for info_item in info_data:
                        if (isinstance(info_item, dict) and 
                            "ref_image_id" in info_item and 
                            normalize_ref_id(info_item["ref_image_id"]) == ref_id):
                            new_item = info_item.copy()
                            break
                
                # Se non trovato in info, crea un elemento base
                if new_item is None:
                    new_item = {
                        "ref_image_id": int(ref_id) if ref_id.isdigit() else ref_id,
                        "relative_cap": "",
                        "tgt_image_id": ""
                    }
                
                # Aggiungi le descrizioni OPT mantenendo il campo multi_caption_dam
                new_item["multi_caption_dam"] = opt_mapping[ref_id]
                dam_data.append(new_item)
                added_count += 1
                print(f"Aggiunto ref_image_id {ref_id}")
    
    print(f"Aggiunti {added_count} elementi mancanti")
    
    # 3. Salva il file corretto
    try:
        output_path = "./data/files/laion_combined_dam_multi_fixed.json"
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(dam_data, file, indent=2, ensure_ascii=False)
        print(f"File corretto salvato in: {output_path}")
        
        # Statistiche finali
        total_fixed = fixed_count + added_count
        print(f"\n=== RIEPILOGO ===")
        print(f"Elementi con errori corretti: {fixed_count}/19")
        print(f"Elementi mancanti aggiunti: {added_count}/21")
        print(f"Totale interventi: {total_fixed}/40")
        
    except Exception as e:
        print(f"Errore nel salvataggio del file: {e}")

def verify_fixes():
    """Verifica che le correzioni siano state applicate correttamente"""
    fixed_data = load_json_file("./data/files/laion_combined_dam_multi_fixed.json")
    if fixed_data is None:
        return
    
    error_count = 0
    fixed_ids_verified = set()
    added_ids_verified = set()
    
    if isinstance(fixed_data, list):
        for item in fixed_data:
            if isinstance(item, dict) and "ref_image_id" in item and "multi_caption_dam" in item:
                ref_id = normalize_ref_id(item["ref_image_id"])
                
                # Controlla se ci sono ancora errori
                multi_caption = item["multi_caption_dam"]
                if isinstance(multi_caption, list):
                    for description in multi_caption:
                        if isinstance(description, str) and "Error generating description with" in description:
                            error_count += 1
                            print(f"ERRORE: ref_image_id {ref_id} ha ancora descrizioni con errori")
                
                # Verifica quali ID sono stati corretti/aggiunti
                if ref_id in [
                    '114023', '165405', '16554', '217690', '2424', '268129', '390135', 
                    '41850', '42650', '43779', '537150', '647006', '71953', '72523', 
                    '73443', '81831', '82860', '844066', '881829'
                ]:
                    fixed_ids_verified.add(ref_id)
                
                if ref_id in [
                    '105668', '114255', '115975', '115981', '116321', '367924', '517184', 
                    '6604', '67627', '72392', '80113', '81123', '827365', '860036', 
                    '89194', '90210', '93529', '95895', '95923', '95944', '98180'
                ]:
                    added_ids_verified.add(ref_id)
    
    print(f"\n=== VERIFICA ===")
    print(f"Descrizioni con errori rimanenti: {error_count}")
    print(f"ID con errori corretti verificati: {len(fixed_ids_verified)}/19")
    print(f"ID mancanti aggiunti verificati: {len(added_ids_verified)}/21")

# Esegui la correzione
print("=== INIZIO CORREZIONE FILE DAM ===")
fix_dam_errors_and_add_missing()

print("\n=== VERIFICA DELLE CORREZIONI ===")
verify_fixes()