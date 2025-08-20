from torch.utils.data import Dataset
import json 
import PIL 
from PIL import Image 
from PIL import ImageFile
import os 
#data_file_path = "./ZS-CIR/ZS-CIR/data" # originale
#data_file_path = "./CapZCIR/data" #mod
data_file_path = "./data" #mod - relative path
Image.MAX_IMAGE_PIXELS = 2300000000
ImageFile.LOAD_TRUNCATED_IMAGES = True 


class LaionDataset_Combined(Dataset):
    def __init__(self, split: str, preprocess: callable):
        self.preprocess = preprocess
        self.split = split

        if split not in ['train']:
            raise ValueError("split should be in ['train']")

        #self.img_path_prefix = "./ZS-CIR/laion_cir_combined/" # originale
        #self.img_path_prefix = "./CapZCIR/data/datasets/laion_cir_combined/" #mod
        self.img_path_prefix = "./data/datasets/laion_cir_combined/" #mod
        # self.img_path_prefix = "./COCO2014/train2014/"
        # # qua dentro ci devo mettere le mie caption generate da me: laion_combined_info.json
        #with open(data_file_path + "/files/laion_combined_info.json") as f: # qua dentro ci devo mettere le mie caption generate da me
        with open(data_file_path + "/files/laion_combined_opt_laion_combined_multi.json") as f: #mod
            self.triplets = json.load(f)

        print(f"Laion {split} dataset initialized")

    def __getitem__(self, index):

        reference_image = f"{str(self.triplets[index]['ref_image_id']).zfill(7)}.png"
        # La seguente è da commentare se si usa solo 1 capiton di quelle originali
        reference_img_texts = [str(x) for x in self.triplets[index]["multi_caption_opt"]] # questo serve se aggiungo le 15 caption
        relative_caption = self.triplets[index]['relative_cap']
        target_image = f"{str(self.triplets[index]['tgt_image_id']).zfill(7)}.png"

        reference_image_path = self.img_path_prefix + reference_image
        reference_image = PIL.Image.open(reference_image_path)
        if reference_image.mode == 'RGB':
            reference_image = reference_image.convert('RGB')
        else:
            reference_image = reference_image.convert('RGBA')
        reference_image = self.preprocess(reference_image)
        target_image_path = self.img_path_prefix + target_image
        target_image = PIL.Image.open(target_image_path)
        if target_image.mode == 'RGB':
            target_image = target_image.convert('RGB')
        else:
            target_image = target_image.convert('RGBA')
        target_image = self.preprocess(target_image)
        return reference_img_texts, target_image, relative_caption # originale
        #return reference_image, target_image, relative_caption # mod, ma originale che usa 1 caption

    def __len__(self):
        return len(self.triplets)