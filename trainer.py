from tqdm import tqdm
import torch 
from torch.utils.data import DataLoader 
import numpy as np 
from statistics import mean
import wandb
from utils import set_train_bar_description, update_train_running_results, extract_index_features, collate_fn

torch.multiprocessing.set_start_method('spawn', force=True)

class Trainer():

    def __init__(self, cfg, model, train_dataloader, optimizer, scheduler, criterion, classic_val_dataset, relative_val_dataset, **kwargs): #rank,
        self.num_epochs = cfg.num_epochs
        self.dataset = cfg.dataset
        if self.dataset == 'fiq':
            self.idx_to_dress_mapping = kwargs['idx_to_dress_mapping']
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = cfg.device
        self.use_amp = cfg.use_amp
        self.criterion = criterion 
        self.encoder = cfg.encoder 
        self.classic_val_dataset = classic_val_dataset
        self.relative_val_dataset = relative_val_dataset
        self.validation_frequency = cfg.validation_frequency
        self.save_path = cfg.save_path
        self.batch_size = cfg.batch_size
        self.epochs=cfg.num_epochs


        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        if self.encoder == 'text' or self.encoder == 'neither':
            self.store_val_features = kwargs



    def train(self):
        best_score = 0
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)

            if epoch % self.validation_frequency == 0:

                results_dict = {}
                if self.dataset == 'cirr':
                    results = self.eval_cirr()
                    group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results
                    results_dict = {
                        'group_recall_at1': group_recall_at1,
                        'group_recall_at2': group_recall_at2,
                        'group_recall_at3': group_recall_at3,
                        'recall_at1': recall_at1,
                        'recall_at5': recall_at5,
                        'recall_at10': recall_at10,
                        'recall_at50': recall_at50,
                        'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                        'arithmetic_mean': mean(results),
                    }
                    print('recall_inset_top1_correct_composition', group_recall_at1)
                    print('recall_inset_top2_correct_composition', group_recall_at2)
                    print('recall_inset_top3_correct_composition', group_recall_at3)
                    print('recall_top1_correct_composition', recall_at1)
                    print('recall_top5_correct_composition', recall_at5)
                    print('recall_top10_correct_composition', recall_at10)
                    print('recall_top50_correct_composition', recall_at50)

                elif self.dataset == 'fiq':
                    results10, results50 = self.eval_fiq()
                    for i in range(len(results10)):
                        results_dict[f'{self.idx_to_dress_mapping[i]}_recall_at10'] = results10[i]
                        results_dict[f'{self.idx_to_dress_mapping[i]}_recall_at50'] = results50[i]
                        print(f'{self.idx_to_dress_mapping[i]}_recall_at10: {results10[i]}')
                        print(f'{self.idx_to_dress_mapping[i]}_recall_at50: {results50[i]}')
                    print('average_recall_at10', mean(results10))
                    print('average_recall_at50', mean(results50))

                    results_dict.update({
                        'average_recall_at10': mean(results10),
                        'average_recall_at50': mean(results50),
                        'average_recall': (mean(results10) + mean(results50)) / 2
                    })

                elif self.dataset == 'circo':
                    results = self.eval_circo()
                    map_at5, map_at10, map_at25, map_at50, recall_at5, recall_at10, recall_at25, recall_at50 = results
                    results_dict = {
                        'circo_map_at5': map_at5,
                        'circo_map_at10': map_at10,
                        'circo_map_at25': map_at25,
                        'circo_map_at50': map_at50,
                        'circo_recall_at5': recall_at5,
                        'circo_recall_at10': recall_at10,
                        'circo_recall_at25': recall_at25,
                        'circo_recall_at50': recall_at50,
                    }
                    print('circo_map_at5', map_at5)
                    print('circo_map_at10', map_at10)
                    print('circo_map_at25', map_at25)
                    print('circo_map_at50', map_at50,)
                    print('circo_recall_at5', recall_at5)
                    print('circo_recall_at10', recall_at10)
                    print('circo_recall_at25', recall_at25)
                    print('circo_recall_at50', recall_at50)

                wandb.log(results_dict)
                if self.dataset == 'cirr':
                    score= mean(results)
                elif self.dataset == 'fiq':
                    score = results_dict['average_recall']
                elif self.dataset == 'circo':
                    score= mean(results)
                if score > best_score:
                    best_score = score
                    self.save_checkpoint(self.save_path)

    def train_epoch(self, epoch):
        self.model.train()
        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
        train_bar = tqdm(self.train_dataloader, ncols=150)
        iters = len(train_bar)
        for idx, batch in enumerate(train_bar):
            reference_img_texts, target_images, captions = batch
            reference_img_texts1 = np.array(reference_img_texts).T.tolist()
            images_in_batch = target_images.size(0)
            self.optimizer.zero_grad()
            target_images = target_images.to(self.device, non_blocking=True)

            if not self.use_amp:
                logits = self.model(captions, reference_img_texts1, target_images)
                ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=self.device)
                loss = self.criterion(logits, ground_truth)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optimizer.step()
                self.scheduler.step(epoch + idx / iters)
            else:
                with torch.cuda.amp.autocast():
                    logits = self.model(captions, reference_img_texts1, target_images)
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=self.device)
                    loss = self.criterion(logits, ground_truth)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.scaler.step(self.optimizer)
                self.scheduler.step(epoch + idx / iters)
                self.scaler.update()

            update_train_running_results(train_running_results, loss, images_in_batch)
            set_train_bar_description(train_bar, epoch, self.num_epochs, train_running_results)
        # wandb to log
            train_epoch_loss = float(train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            wandb.log({'train_epoch_loss': train_epoch_loss})

    def get_val_index_features(self, index=None):

        with torch.no_grad():
            if (self.encoder == 'both' or self.encoder == 'image') and self.dataset == 'cirr':
                val_index_features, val_index_names, _ = extract_index_features(self.classic_val_dataset, self.model, return_local=False)
            elif self.dataset == 'cirr':
                val_index_features, val_index_names, _ = self.store_val_features['val_index_features'], self.store_val_features['val_index_names'], self.store_val_features['val_total_index_features']
            elif (self.encoder == 'both' or self.encoder == 'image') and self.dataset == 'fiq':
                val_index_features, val_index_names, _ = extract_index_features(self.classic_val_dataset[index], self.model, return_local=False)
            elif self.dataset == 'fiq':
                val_index_features, val_index_names, _ = self.store_val_features['val_index_features'][index], self.store_val_features['val_index_names'][index], self.store_val_features['val_total_index_features'][index]
            elif (self.encoder == 'both' or self.encoder == 'image') and self.dataset =='circo':
                val_index_features, val_index_names, _ = extract_index_features(self.classic_val_dataset, self.model, return_local=False)
            elif self.dataset == 'circo':
                val_index_features, val_index_names, _ = self.store_val_features['val_index_features'], self.store_val_features['val_index_names'], self.store_val_features['val_total_index_features']
        return val_index_features, val_index_names, _


    def eval_cirr(self):
        self.model.eval()
        val_index_features, val_index_names, _ = self.get_val_index_features()
        results = self.compute_cirr_val_metrics(val_index_names, val_index_features)
        return results 


    def eval_fiq(self):
        self.model.eval()
        recalls_at10 = []
        recalls_at50 = []
        for idx in self.idx_to_dress_mapping:
            val_index_features, val_index_names, val_index_total_features = self.get_val_index_features(index=idx)
            recall_at10, recall_at50 = self.compute_fiq_val_metrics(val_index_names, val_index_features, val_index_total_features, idx)
            recalls_at10.append(recall_at10)
            recalls_at50.append(recall_at50)
        results_dict = {}
        for i in range(len(recalls_at10)):
            results_dict[f'{self.idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
            results_dict[f'{self.idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
        wandb.log(results_dict)
        return recalls_at10, recalls_at50

    def eval_circo(self):
        self.model.eval()
        val_index_features, val_index_names, _ = self.get_val_index_features()
        results = self.compute_circo_val_metrics(val_index_names, val_index_features)
        return results

    def get_val_dataloader(self, index=None):
        if index == None:
            dataset = self.relative_val_dataset
        else:
            dataset = self.relative_val_dataset[index]

        # Aggiunto: DEBUG
        print(f"DEBUG: Validation dataset size: {len(dataset)}")
        
        if len(dataset) == 0:
            print("ERROR: Validation dataset is empty!")
            print("Checking dataset paths and files...")
            
            # Prova ad accedere al primo elemento per vedere se ci sono errori
            try:
                sample = dataset[0]
                print(f"DEBUG: First sample: {sample}")
            except Exception as e:
                print(f"ERROR accessing dataset[0]: {e}")
        # Fine aggiunta debug
        #relative_val_loader = DataLoader(dataset=dataset, batch_size=16, num_workers=8, pin_memory=True,collate_fn=collate_fn) #originale
        relative_val_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True, collate_fn=collate_fn) #modificato in modo da prendere il batch_size giusto
        return relative_val_loader

    def compute_fiq_val_metrics(self, val_index_names, val_index_features, val_total_index_features, index):
        relative_val_loader = self.get_val_dataloader(index)
        target_names = []
        predicted_features_list = []

        for batch_reference_names, batch_reference_image_texts, batch_target_names, captions in tqdm(relative_val_loader):
            batch_reference_image_texts=np.array(batch_reference_image_texts).T.tolist()
            flattened_captions: list = np.array(captions).T.flatten().tolist()
            input_captions = [
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
                i in range(0, len(flattened_captions), 2)]
            with torch.no_grad():
                # reference_images = batch_reference_image.to(self.device)
                batch_predicted_features = self.model.combine_features(batch_reference_image_texts, input_captions)
                predicted_features_list.append(batch_predicted_features / batch_predicted_features.norm(dim=-1, keepdim=True))
            
            target_names.extend(batch_target_names)
        predicted_features = torch.cat(predicted_features_list, dim=0)
        val_index_features = val_index_features / val_index_features.norm(dim=-1, keepdim=True)

        distances = 1 - predicted_features @ val_index_features.T
        results = self.compute_results(distances, val_index_names, target_names)
        
        return results 
            

    def compute_cirr_val_metrics(self, val_index_names, val_index_features):

        relative_val_loader = self.get_val_dataloader()

        # Aggiunto: Controlla se il dataloader è vuoto
        if len(relative_val_loader) == 0:
            print("ERROR: Validation dataloader is empty!")
            return 0, 0, 0, 0, 0, 0, 0
        # Fine aggiunta debug

        target_names = []
        group_members = []
        reference_names = []
        predicted_features_list = []

        print(f"DEBUG: Starting validation with {len(relative_val_loader)} batches") # Aggiunto debug

        for batch_reference_names, batch_reference_img_texts, batch_target_names, captions, batch_group_members in tqdm(relative_val_loader):
            batch_group_members = np.array(batch_group_members).T.tolist()
            batch_reference_img_texts=np.array(batch_reference_img_texts).T.tolist()
            with torch.no_grad():
                # reference_images = batch_reference_img.to(self.device)
                batch_predicted_features = self.model.combine_features(batch_reference_img_texts, captions)
                predicted_features_list.append(batch_predicted_features / batch_predicted_features.norm(dim=-1, keepdim=True))

            target_names.extend(batch_target_names)
            group_members.extend(batch_group_members)
            reference_names.extend(batch_reference_names)
        
        predicted_features = torch.cat(predicted_features_list, dim=0)

        val_index_features = val_index_features / val_index_features.norm(dim=-1, keepdim=True)

        distances = 1 - predicted_features @ val_index_features.T

        results = self.compute_results(distances, val_index_names, target_names, reference_names, group_members)
        
        return results

    def compute_circo_val_metrics(self, val_index_names, val_index_features):
        relative_val_loader = self.get_val_dataloader()
        predicted_features_list = []
        target_names_list = []
        gts_img_ids_list = []
        reference_names_list = []

        for reference_img_texts, reference_names, target_names, target_imgs, relative_captions, gt_img_ids in tqdm(relative_val_loader):
            gt_img_ids = np.array(gt_img_ids).T.tolist()
            reference_img_texts=np.array(reference_img_texts).T.tolist()

            with torch.no_grad():
                batch_predicted_features = self.model.combine_features(reference_img_texts, relative_captions)
                predicted_features_list.append(
                    batch_predicted_features / batch_predicted_features.norm(dim=-1, keepdim=True))

            target_names_list.extend(target_names)
            gts_img_ids_list.extend(gt_img_ids)
            reference_names_list.extend(reference_names)

        predicted_features = torch.cat(predicted_features_list, dim=0)

        val_index_features = val_index_features / val_index_features.norm(dim=-1, keepdim=True)

        results = self.compute_results_circo(predicted_features, val_index_names, val_index_features, target_names_list,
                                             reference_names_list, gts_img_ids_list)

        return results

    def compute_results(self, distances, val_index_names,  target_names, reference_names=None, group_members=None):
        sorted_indices = torch.argsort(distances, dim=-1).cpu()
        sorted_index_names = np.array(val_index_names)[sorted_indices]

        if reference_names == None:
            labels = torch.tensor(
                sorted_index_names == np.repeat(np.array(target_names), len(val_index_names)).reshape(len(target_names), -1))

            recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
            recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

            return recall_at10, recall_at50

        elif reference_names != None:
            reference_mask = torch.tensor(
                sorted_index_names != np.repeat(np.array(reference_names), len(val_index_names)).reshape(len(target_names), -1))
            sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                            sorted_index_names.shape[1] - 1)

            labels = torch.tensor(
                sorted_index_names == np.repeat(np.array(target_names), len(val_index_names) - 1).reshape(len(target_names), -1))

            group_members = np.array(group_members)
        
            group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
            group_labels = labels[group_mask].reshape(labels.shape[0], -1)

            assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
            assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

            # Compute the metrics
            recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
            recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
            recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
            recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
            group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
            group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
            group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

            return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50

    def compute_results_circo(self, predicted_features, val_index_names, val_index_features, target_names,
                              reference_names, gts_img_ids):

        ap_at5 = []
        ap_at10 = []
        ap_at25 = []
        ap_at50 = []

        recall_at5 = []
        recall_at10 = []
        recall_at25 = []
        recall_at50 = []

        index_features = val_index_features
        predicted_features = predicted_features

        for predicted_feature, target_name, gt_img_ids in tqdm(zip(predicted_features, target_names, gts_img_ids)):
            gt_img_ids = np.array(gt_img_ids)[
                np.array(gt_img_ids) != '']  # remove trailing empty strings added for collate_fn
            similarity = predicted_feature @ index_features.T
            sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
            sorted_index_names = np.array(val_index_names)[sorted_indices]
            map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
            precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
            precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position
            # Compute the metrics
            ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
            ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
            ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
            ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

            assert target_name == gt_img_ids[0], f"Target name not in GTs {target_name} {gt_img_ids}"
            single_gt_labels = torch.tensor(sorted_index_names == target_name)
            recall_at5.append(float(torch.sum(single_gt_labels[:5])))
            recall_at10.append(float(torch.sum(single_gt_labels[:10])))
            recall_at25.append(float(torch.sum(single_gt_labels[:25])))
            recall_at50.append(float(torch.sum(single_gt_labels[:50])))

        map_at5 = np.mean(ap_at5) * 100
        map_at10 = np.mean(ap_at10) * 100
        map_at25 = np.mean(ap_at25) * 100
        map_at50 = np.mean(ap_at50) * 100
        recall_at5 = np.mean(recall_at5) * 100
        recall_at10 = np.mean(recall_at10) * 100
        recall_at25 = np.mean(recall_at25) * 100
        recall_at50 = np.mean(recall_at50) * 100

        return map_at5, map_at10, map_at25, map_at50, recall_at5, recall_at10, recall_at25, recall_at50

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)




        