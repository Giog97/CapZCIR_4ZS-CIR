import torch
from dataclasses import dataclass

@dataclass
class Config:
    dropout: float = 0.1
    num_layers: int = 2
    model_name: str = 'blip' # [blip, clip-Vit-B/32, clip-Vit-L/14]
    device: torch.device = torch.device('cuda')
    batch_size: int = 4 # init 16 you can adjust it according to your GPU memory
    encoder: str = 'text' # ['neither', 'text', 'both']
    laion_type: str = 'laion_combined' # ['laion_combined', 'laion_template', 'laion_llm', 'laion_coco_combined', lasco] choose different dataset
    transform: str = 'targetpad'
    target_ratio: float = 1.25
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    adam_epsilon: float = 1e-8
    num_epochs: int = 15 # 100 inizialmente
    save_best: bool = True
    use_amp: bool = True
    load: str = 'pretrained' #[pretrained, trained] #orginal:trained --> load: 'pretrained': Carica solo i pesi pre-addestrati di BLIP/CLIP e addestra da zero- load: 'trained': Cerca di caricare un modello già addestrato dal percorso specificato in eval_load_path
    validation_frequency: int = 3 # fa la validazione dopo ogni epoca se impostato =1 (originale 1)
    #data_root: str = './CapZCIR/data'  # Add this line to specify your data root #AGGIUNTO
    #data_root: str = './data'  # Add this line to specify your data root #AGGIUNTO
    comment: str = "cirr_train_15epoch_blipbase" # nome che viene dato al modello, originale: "cirr_text_our_2L8H_blipbase"
    dataset: str="cirr" # ['fiq', 'cirr','circo']
    #save_path_prefix ='./ZS-CIR/new' # originale
    #save_path_prefix ='./CapZCIR/new' # mod
    save_path_prefix ='./new' # mod
    # eval related
    #eval_load_path: str="./ZS-CIR/new/2025-03-31-16-01-21_cirr" # originale
    #eval_load_path: str="./CapZCIR/new/2025-03-31-16-01-21_cirr" #mod
    eval_load_path: str="./new/2025-03-31-16-01-21_cirr"
    submission_name: str='circo_our_2L8H_base'

