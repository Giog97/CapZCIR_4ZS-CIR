from torch.utils.data import Dataset
import json
import PIL
from PIL import Image
from PIL import ImageFile
import os
#data_file_path = "./ZS-CIR/ZS-CIR/data" # originale
#data_file_path = "./CapZCIR/data" #mod
data_file_path = "./data" #mod
Image.MAX_IMAGE_PIXELS = 2300000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Laion_COCO_Dataset_Combined(Dataset):
    def __init__(self, split: str, preprocess: callable):
        self.preprocess = preprocess
        self.split = split

        if split not in ['train']:
            raise ValueError("split should be in ['train']")

        #self.image_path_prefix = "./ZS-CIR/laion_cir_combined/" # originale
        #self.image_path_prefix = "./CapZCIR/data/datasets/laion_cir_combined/" #mod
        self.image_path_prefix = "./data/datasets/laion_cir_combined/" #mod
        #self.img_path_prefix = "./COCO2014" # commentato rispetto all'originale
        with open(data_file_path + "/files/laion_combined.json") as f:
            self.triplets = json.load(f)

        print(f"Laion_coco {split} dataset initialized")

    def __getitem__(self, index):
        reference_image = str(self.triplets[index]['ref_image_id'])
        relative_caption = self.triplets[index]['relative_cap']
        reference_image_text= self.triplets[index]["multi_caption_opt"]
        #target_image= str(self.triplets[index]['tgt_image_id'])
        # if reference_image.startswith("train2014"):
        #     reference_image_path = self.img_path_prefix + '/' + reference_image
        # else:
        reference_image = f"{str(self.triplets[index]['ref_image_id']).zfill(7)}.png"
        reference_image_path = self.image_path_prefix + '/' + reference_image
        # if target_image.startswith("train2014"):
        #     target_image_path = self.img_path_prefix + '/' +target_image
        # else:
        target_image = f"{str(self.triplets[index]['tgt_image_id']).zfill(7)}.png"
        target_image_path = self.image_path_prefix + '/' + target_image

        reference_image ="" #PIL.Image.open(reference_image_path)
        # if reference_image.mode == 'RGB':
        #     reference_image = reference_image.convert('RGB')
        # else:
        #     reference_image = reference_image.convert('RGBA')
        # reference_image = self.preprocess(reference_image)

        target_image = PIL.Image.open(target_image_path)
        if target_image.mode == 'RGB':
            target_image = target_image.convert('RGB')
        else:
            target_image = target_image.convert('RGBA')
        target_image = self.preprocess(target_image)
        return reference_image, reference_image_text, target_image, relative_caption

    def __len__(self):
        return len(self.triplets)
