import torch
from dataclasses import dataclass

@dataclass
class Config:
    dropout: float = 0.1 # potrei provare ad alzarlo per ridurre overfitting
    num_layers: int = 2
    model_name: str = 'blip' # [blip, clip-Vit-B/32, clip-Vit-L/14]
    device: torch.device = torch.device('cuda')
    batch_size: int = 16 # era 8  # init 16 you can adjust it according to your GPU memory # Con 16 non gira sulle Dream Machine
    encoder: str = 'text' # ['neither', 'text', 'both']
    laion_type: str = 'laion_combined' # ['laion_combined', 'laion_template', 'laion_llm', 'laion_coco_combined', lasco] choose different dataset
    transform: str = 'targetpad'
    target_ratio: float = 1.25
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    adam_epsilon: float = 1e-8
    num_epochs: int = 50 # 100 originale, inizialmente usato 15
    save_best: bool = True
    use_amp: bool = True
    #load: str = 'pretrained' #[pretrained, trained] #orginal:trained --> load: 'pretrained': Carica solo i pesi pre-addestrati di BLIP/CLIP e addestra da zero- load: 'trained': Cerca di caricare un modello già addestrato dal percorso specificato in eval_load_path
    load: str = 'trained' #[pretrained, trained] #orginal:trained --> load: 'pretrained': Carica solo i pesi pre-addestrati di BLIP/CLIP e addestra da zero- load: 'trained': Cerca di caricare un modello già addestrato dal percorso specificato in eval_load_path
    validation_frequency: int = 1 # Fa la validazione dopo ogni x epoche. Se impostato =1 (originale 1) lo fa dopo ogni epoche, il che è buono per ottenre best pesi
    #comment: str = "cirr_train_50epoch_blipbase" # nome che viene dato al modello su W&B, originale: "cirr_text_our_2L8H_blipbase"
    comment: str = "cirr_train_50epoch_bliplarge" # nome che viene dato al modello su W&B, originale: "cirr_text_our_2L8H_blipbase"
    #comment: str = "cirr_train_50epoch_vitb32" # nome che viene dato al modello su W&B, originale: "cirr_text_our_2L8H_blipbase"
    dataset: str="circo" # ['fiq', 'cirr','circo']
    save_path_prefix ='./new' # mod
    
    # eval related
    eval_load_path: str="./new/2025-10-08-DAMtv_cirr_train_50epoch_blipbase_batch16_2textencoder_best_arithmetic" # path dei pesi da caricare
    #submission_name: str='cirr_our_DAMtv_cirr_train_50epoch_blipbase_best_arithmetic_16batch' # nome che verrà dato alla submission su CIRCO
    #submission_name: str='cirr_our_DAMtv_cirr_train_50epoch_bliplarge_best_arithmetic_16batch'
    submission_name: str='circo_our_DAMtv_cirr_train_50epoch_blipbase__16batch_2textencoder_best_arithmetic'


