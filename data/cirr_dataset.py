from torch.utils.data import Dataset
import json 
import PIL 
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 2300000000


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        #self.cirr_path_prefix = "./datasets" #originale
        #self.cirr_path_prefix = "./CapZCIR/data/datasets" # mod
        self.cirr_path_prefix = "./data/datasets" # mod
        self.preprocess = preprocess
        self.mode = mode
        self.split = split # ['test', 'train', 'val']
        if self.split == 'test_train':
            split = 'train'

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val'")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        # with open(f'{self.cirr_path_prefix}/CIRR/cirr/captions/cap.rc2.{split}.json') as f:
        #     self.triplets = json.load(f)

        # with open('./ZS-CIR/ZS-CIR/data/files/val_cirr.json') as f:
        #     self.triplets = json.load(f)

        #with open('./ZS-CIR/ZS-CIR/data/files/cap.rc2.test1.json') as f: #originale
        #    self.triplets = json.load(f)
        #with open('./CapZCIR/data/files/cap.rc2.test1.json') as f: # mod
        #with open('./data/files/cap.rc2.test1.json') as f: # mod --> usa cap.rc2.test1.json di Pavan ma NON è l'originale di CIRR
        #with open('./data/files/cap.rc2.test1_fixed.json') as f: # mod --> usa cap.rc2.test1.json di Pavan ma NON è l'originale di CIRR
        #with open(f'{self.cirr_path_prefix}/CIRR/cirr/captions/cap.rc2.{split}.json') as f: # mod --> usa cap.rc2.test1.json l'originale di CIRR
        
        # with open('./data/files/val_cirr_opt_laion_combined_multi.json') as f: #preso da Pavan

        # ==== Parte Nuova ====
        # I seguenti file sono stati generati per contenere il campo 'multi_caption_dam' 
        # In questo campo sono contenute tutte le descrizioni ottenute o con BLIP o con DAM in base al file

        # VAL_BLIP desc: Seguente codice prende le descrizioni di validation CIRR ottenute con BLIP
        #with open('./data/files/scarti/val_cirr_opt_laion_combined_multi_fixed.json') as f: #rinominato per funzionare con campo 'multi_caption_dam'
        #     self.triplets = json.load(f)
        
        # TEST_BLIP desc:  Seguente codice prende le descrizioni di test1 CIRR ottenute con BLIP
        #with open('./data/files/scarti/cap.rc2.test1_fixed.json') as f: # mod --> usa cap.rc2.test1.json di Pavan ma modifato per funzionare con campo 'multi_caption_dam'
        #    self.triplets = json.load(f)

        # VAL_DAM desc: Seguente codice prende le descrizioni di validation CIRR ottenute con DAM griglie multilivello
        #with open('./data/files/cap.rc2.val_dam.json') as f: # mod --> usa cap.rc2.val.json di Pavan ma con descrizioni DAM
        #    self.triplets = json.load(f)

        # TEST_DAM desc:  Seguente codice prende le descrizioni di test1 CIRR ottenute con DAM griglie multilivello
        with open('./data/files/cap.rc2.test1_dam.json') as f: # mod --> usa cap.rc2.test1.json di Pavan ma con descrizioni DAM
            self.triplets = json.load(f)

        # ---- parte precedente carica il file delle descrizioni per fare validation o test

        # get a mapping from image name to relative path
        #with open(f'{self.cirr_path_prefix}/CIRR/cirr/image_splits/split.rc2.{split}.json') as f: #originale
        with open(f'{self.cirr_path_prefix}/CIRR/cirr/image_splits/split.rc2.{split}.json') as f: #mod
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

        #print(f"DEBUG: CIRR path prefix: {self.cirr_path_prefix}")
        #print(f"DEBUG: Number of triplets: {len(self.triplets)}")
        #print(f"DEBUG: Number of image mappings: {len(self.name_to_relpath)}")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption'].lower()

                if self.split == 'train':
                    #print(f"DEBUG: PASSA DALLO SPLIT TRAIN: {self.split}")
                    #reference_image_path = f"{self.cirr_path_prefix}/CIRR/" + self.name_to_relpath[reference_name][2:] #originale
                    reference_image_path = f"{self.cirr_path_prefix}/CIRR/" + self.name_to_relpath[reference_name][2:] #mod
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert('RGB'))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = f"{self.cirr_path_prefix}/CIRR/" + self.name_to_relpath[target_hard_name][2:]
                    target_image = self.preprocess(PIL.Image.open(target_image_path).convert("RGB"))
                    return reference_image, target_image, rel_caption

                elif self.split == 'val':
                    #print(f"DEBUG: PASSA DALLO SPLIT VALIDATION: {self.split}")
                    #reference_image_path = f"{self.cirr_path_prefix}/CIRR/" + self.name_to_relpath[reference_name][2:] #originale
                    reference_image_path = f"{self.cirr_path_prefix}/CIRR/" + self.name_to_relpath[reference_name][2:] #mod
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert('RGB'))
                    #reference_img_texts = [str(x) for x in self.triplets[index]["multi_caption_opt"]] #originale Pavan
                    reference_img_texts = [str(x) for x in self.triplets[index]["multi_caption_dam"]]
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, reference_img_texts, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    #print(f"DEBUG: PASSA DALLO SPLIT TEST: {self.split}")
                    reference_image_path = f"{self.cirr_path_prefix}/CIRR/" + self.name_to_relpath[reference_name][2:]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert('RGB')) # non so come mai ma era commentato nell'originale
                    #reference_img_texts = [str(x) for x in self.triplets[index]["multi_caption_opt"]] #originale Pavan
                    reference_img_texts = [str(x) for x in self.triplets[index]["multi_caption_dam"]] # quando farò le mie descrizioni
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, reference_img_texts, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = f"{self.cirr_path_prefix}/CIRR/" + self.name_to_relpath[image_name][2:]
                im = PIL.Image.open(image_path).convert("RGB")
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)  # Number of query triplets
        elif self.mode == 'classic':
            return len(self.name_to_relpath) # Number of images in gallery
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
