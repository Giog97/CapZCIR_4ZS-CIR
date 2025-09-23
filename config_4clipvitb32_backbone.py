import torch
from dataclasses import dataclass

@dataclass
class Config:
    dropout: float = 0.3 # 0.1 mod per migliorare train CLIP
    num_layers: int = 2
    model_name: str = 'clip-Vit-B/32' # [blip, clip-Vit-B/32, clip-Vit-L/14]
    device: torch.device = torch.device('cuda')
    batch_size: int = 16 # 4 provato e funziona # init 16 you can adjust it according to your GPU memory # Con 16 non gira sulle Dream Machine
    encoder: str = 'text' # ['neither', 'text', 'both']
    laion_type: str = 'laion_combined' # ['laion_combined', 'laion_template', 'laion_llm', 'laion_coco_combined', lasco] choose different dataset
    transform: str = 'targetpad'
    target_ratio: float = 1.25
    learning_rate: float = 1e-5 # 1e-4 mod per migliorare train CLIP
    weight_decay: float = 0.01 # 0.05 mod per migliorare train CLIP
    adam_epsilon: float = 1e-8
    num_epochs: int = 50 # 100 originale, inizialmente usato 15
    save_best: bool = True
    use_amp: bool = False # True mod per migliorare train CLIP (overflow FP16)
    load: str = 'pretrained' #[pretrained, trained] #orginal:trained --> load: 'pretrained': Carica solo i pesi pre-addestrati di BLIP/CLIP e addestra da zero- load: 'trained': Cerca di caricare un modello già addestrato dal percorso specificato in eval_load_path
    #load: str = 'trained' #[pretrained, trained] #orginal:trained --> load: 'pretrained': Carica solo i pesi pre-addestrati di BLIP/CLIP e addestra da zero- load: 'trained': Cerca di caricare un modello già addestrato dal percorso specificato in eval_load_path
    validation_frequency: int = 1 # Fa la validazione dopo ogni x epoche. Se impostato =1 (originale 1) lo fa dopo ogni epoche, il che è buono per ottenre best pesi
    #comment: str = "cirr_train_15epoch_blipbase" # nome che viene dato al modello su W&B, originale: "cirr_text_our_2L8H_blipbase"
    comment: str = "cirr_train_50epoch_vitb32" # nome che viene dato al modello su W&B, originale: "cirr_text_our_2L8H_blipbase"
    dataset: str="cirr" # ['fiq', 'cirr','circo']
    #save_path_prefix ='./ZS-CIR/new' # originale
    #save_path_prefix ='./CapZCIR/new' # mod
    save_path_prefix ='./new' # mod
    
    # eval related
    #eval_load_path: str="./ZS-CIR/new/2025-03-31-16-01-21_cirr" # originale
    #eval_load_path: str="./CapZCIR/new/2025-03-31-16-01-21_cirr" #mod
    #eval_load_path: str="./new/2025-03-31-16-01-21_cirr" # funzionante
    eval_load_path: str="./new/2025-09-11-DAMtv_cirr_train_15epoch_blipbase_best_arithmetic"
    #2025-09-04-DAMt_BLIPv_cirr_train_15epoch_blipbase_best_arithmetic
    #2025-09-11-DAMtv_cirr_train_15epoch_blipbase_best_arithmetic
    #submission_name: str='circo_our_2L8H_base' # originale
    submission_name: str='cirr_our_DAMtv_cirr_train_15epoch_blipbase_best_arithmetic' # originale

