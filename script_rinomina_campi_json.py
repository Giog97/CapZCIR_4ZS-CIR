import json

# Percorso del file
#input_file = "./data/files/cap.rc2.test1.json" #file originale di Pavan, in cui abbiamo le descrizioni per fare test generate da blip
input_file = "./data/files/val_cirr_opt_laion_combined_multi.json" #file originale di Pavan, in cui abbiamo le descrizioni per fare validation generate da blip
#output_file = "./data/files/cap.rc2.test1_fixed.json" # file modificato per fare test con Dam
output_file = "./data/files/val_cirr_opt_laion_combined_multi_fixed.json" # file modificato per validation test con Dam

# Carica il JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Itera sugli elementi e rinomina la chiave
for item in data:
    if "multi_caption_opt" in item:
        item["multi_caption_dam"] = item.pop("multi_caption_opt") # sostituisco nome di chiave 'multi_caption_opt' con 'multi_caption_dam'

# Salva il risultato in un nuovo file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Campo rinominato e salvato in", output_file)
