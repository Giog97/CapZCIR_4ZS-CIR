from torch.utils.data import Dataset
from typing import List, Optional, Union, Dict, Literal
import json
import PIL
from PIL import Image
from pathlib import Path

class CIRCODataset(Dataset):
    """
    CIRCO dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions', 'shared_concept',
             'gt_img_ids', 'query_id'] when split == 'val'
            - ['reference_image', 'reference_name', 'relative_captions', 'shared_concept', 'query_id'] when split == test
    """

    def __init__(self,  split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable):
        """
        Args:
            dataset_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        #dataset_path = Path('./datasets/CIRCO') #originale
        #dataset_path = Path('./CapZCIR/data/datasets/CIRCO') #mod
        dataset_path = Path('./data/datasets/CIRCO') #mod
        #data_path = Path('./ZS-CIR/ZS-CIR/ZS-CIR/data/files') #originale
        #data_path = Path('./CapZCIR/data/files') # mod
        data_path = Path('./data/files') # mod
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val','val2']:
            raise ValueError("split should be in ['test','val']")

        # Load COCO images information
        with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        # with open('./ZS-CIR/ZS-CIR/data/files/val.json', "r") as f:
        #     self.annotations: List[dict] = json.load(f)

        #with open('./ZS-CIR/ZS-CIR/circo_test.json', "r") as f: #originale
        with open(f'{self.data_path}/circo_test.json', "r") as f: #mod
            self.annotations: List[dict] = json.load(f)
        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """cd .
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index):
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id]
            if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['query_id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_cap']
            # shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['ref_image_id']).lstrip('0')
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            # reference_img = self.preprocess(PIL.Image.open(reference_img_path).convert('RGB'))
            reference_img = PIL.Image.open(reference_img_path).convert('RGB')
            reference_img_texts= self.annotations[index]["captions"]

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['tgt_image_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path).convert('RGB'))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return reference_img_texts, reference_img_id, target_img_id, target_img, relative_caption, gt_img_ids

            elif self.split == 'test':
                return reference_img_texts, reference_img_id, relative_caption, query_id

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]
            # Preprocess image and return
            img = self.preprocess(PIL.Image.open(img_path).convert('RGB'))
            return img_id, img

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")