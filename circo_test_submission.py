import json
import multiprocessing
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from model.model import ZSCIR
from data.circo_dataset import CIRCODataset
from utils import get_preprocess, extract_index_features, collate_fn
@torch.no_grad()
def circo_generate_test_submission_file(file_name, model, preprocess, device) -> None:
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """

    # Compute the index features
    classic_test_dataset = CIRCODataset('test', 'classic', preprocess)
    index_features, index_names, _ = extract_index_features(classic_test_dataset, model, return_local=False)
    relative_test_dataset = CIRCODataset('test', 'relative', preprocess)

    # Get the predictions dict
    queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, model, index_features, index_names, device=device)

    print(f"Saving CIRR test predictions")
    with open(f"./ZS-CIR/submission/recall_submission_{file_name}.json", 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)


def circo_generate_test_predictions(model, relative_test_dataset: CIRCODataset, device) -> [torch.Tensor, List[List[str]]]:
    """
    Generate the test prediction features for the CIRCO dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=10,
                                      pin_memory=True, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []  
    query_ids_list = []

    # Compute the predictions
    for reference_image_text, refernce_name, relative_captions, query_ids in tqdm(relative_test_loader):
        reference_img_texts = np.array(reference_image_text).T.tolist()
        with torch.no_grad():
           batch_predicted_features= model.combine_features(reference_img_texts, relative_captions)
           predicted_features_list.append(
               batch_predicted_features / batch_predicted_features.norm(dim=-1, keepdim=True))

        query_ids_list.extend(query_ids)

    predicted_features = torch.cat(predicted_features_list, dim=0)

    print('\nEstimated time: {} seconds.\n'.format(model.capzcir_time / len(relative_test_loader)))

    return predicted_features, query_ids_list


def circo_generate_test_dict(relative_test_dataset: CIRCODataset, model, index_features: torch.Tensor,
                             index_names: List[str],  device):
    """
    Generate the test submission dicts for the CIRCO dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, query_ids = circo_generate_test_predictions(model, relative_test_dataset, device)

    # Normalize the features
    index_features = index_features.float().to(device)
    index_features = F.normalize(index_features, dim=-1)

    # Compute the similarity
    similarity = predicted_features @ index_features.T
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                   (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)}

    return queryid_to_retrieved_images

def main():
    cfg = Config()
    model = ZSCIR(cfg)
    device = cfg.device
    model = model.to(device)
    model.load_state_dict(torch.load(cfg.eval_load_path))
    print("the model is loaded")
    if cfg.model_name.startswith("blip"):
        input_dim = 384
    elif cfg.model_name.startswith("clip"):
        input_dim = model.pretrained_model.visual.input_resolution

    preprocess = get_preprocess(cfg, model, input_dim=input_dim)

    model.eval()

    circo_generate_test_submission_file(cfg.submission_name, model, preprocess, device)


if __name__ == '__main__':
    main()
