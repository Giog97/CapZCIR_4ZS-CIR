from torch.utils.data import Dataset
from typing import List
import json 
import PIL 

class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        #self.fiq_path_prefix = "./datasets" # originale
        #self.fiq_path_prefix = "./CapZCIR/data/datasets" # mod
        self.fiq_path_prefix = "./data/datasets" # mod
        self.mode = mode
        self.dress_types = dress_types
        self.split = split

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions) # aggiunto al file .json multi_caption_dam con le descrizioni
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            #with open(f'./FashionIQ/captions/cap.{dress_type}.{split}.json') as f: #originale
            # IL SEGUENTE FILE DOVRà ESSERE IL FILE .JSON CONTENENTE IL FILE ORIGINALE + LE DESCRIZIONI OTTENUTE CON DAM
            #with open(f'{self.fiq_path_prefix}/FashionIQ/captions/cap.{dress_type}.{split}.json') as f: #mod MA ORIGINALE
            #    self.triplets.extend(json.load(f))
            # Con il seguente caricamento del file carichiamo le info per CIR e le descrizioni DAM (di val o test)
            # DESCRIZIONI DAM + BLIP
            with open(f'./data/files/{dress_type}.{split}_dam.json') as f: #mod 
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            #with open(f'./FashionIQ/image_splits/split.{dress_type}.{split}.json') as f: #originale
            with open(f'{self.fiq_path_prefix}/FashionIQ/image_splits/split.{dress_type}.{split}.json') as f: #mod
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions'] # captions = lista di frasi (“relative captions”) fornite da annotatori umani che descrivono le differenze visive fra la candidate image e la target image
                reference_name = self.triplets[index]['candidate'] # candidate = è l’ID di un’immagine di partenza / riferimento, cioè quella che l’utente sta guardando/inserendo come punto di partenza

                if self.split == 'train':
                    #reference_image_path = f'{self.fiq_path_prefix}/FashionIQ/images/{reference_name}.jpg' #originale
                    reference_image_path = f'{self.fiq_path_prefix}/FashionIQ/images/{reference_name}.jpg'
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    target_image_path = f'{self.fiq_path_prefix}/FashionIQ/images/{target_name}.jpg'
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, image_captions

                elif self.split == 'val':
                    #reference_image_path = f'{self.fiq_path_prefix}/FashionIQ/images/{reference_name}.jpg' #originale
                    reference_image_path = f'{self.fiq_path_prefix}/FashionIQ/images/{reference_name}.jpg'
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    # reference_image = PIL.Image.open(reference_image_path).convert('RGB')
                    reference_image_texts=self.triplets[index]["multi_caption_opt"] # "multi_caption_opt" = descrizioni estratte con BLIP
                    #reference_image_texts=self.triplets[index]["multi_caption_dam"] # "multi_caption_dam" = descrizioni estratte con DAM
                    target_name = self.triplets[index]['target']
                    return reference_name, reference_image_texts,target_name, image_captions

                elif self.split == 'test':
                    reference_image_path = f'{self.fiq_path_prefix}/FashionIQ/images/{reference_name}.jpg'
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = f'{self.fiq_path_prefix}/FashionIQ/images/{image_name}.jpg'
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")