# visualize_cirr_comparison_final.py
# Per lanciarlo:
# python visualize_cirr_comparison.py --num_queries 1000 --top_cases 10 --output_dir ./my_comparisons_cirr
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps
import os
import sys
import argparse
from pathlib import Path
import textwrap

# Aggiungi il path ai tuoi modelli
#sys.path.append('./')

from config_test_comparison import Config
from model.model_working_1encoder import ZSCIR as ZSCIR_1t
from model.model_working_2encoder import ZSCIR as ZSCIR_2tv
from utils import get_preprocess, extract_index_features
from data.cirr_dataset import CIRRDataset

class CIRRComparison:
    def __init__(self, config_1t, config_2tv, device='cuda'):
        self.device = device
        self.config_1t = config_1t
        self.config_2tv = config_2tv
        
        # Carica i modelli
        print("Caricamento modello BLIP 1t...")
        self.model_1t = self.load_model(config_1t, ZSCIR_1t)
        
        print("Caricamento modello BLIP 2tv...")
        self.model_2tv = self.load_model(config_2tv, ZSCIR_2tv)
        
        # Setup preprocessing
        if config_1t.model_name.startswith('blip'):
            input_dim = 384
        else:
            input_dim = self.model_1t.pretrained_model.visual.input_resolution
            
        self.preprocess = get_preprocess(config_1t, self.model_1t, input_dim)
        
        # Carica dataset
        print("Caricamento dataset CIRR...")
        self.classic_dataset = CIRRDataset(split='val', mode='classic', preprocess=self.preprocess)
        self.relative_dataset = CIRRDataset(split='val', mode='relative', preprocess=self.preprocess)
        
        # Precompute index features
        print("Precomputing index features...")
        self.index_features, self.index_names, _ = extract_index_features(
            self.classic_dataset, self.model_1t, return_local=False
        )
        
        # Carica i triplets per avere accesso alle informazioni
        with open('./data/files/scarti/val_cirr_opt_laion_combined_multi.json', 'r') as f:
            self.triplets = json.load(f)
            
        with open('./data/datasets/CIRR/cirr/image_splits/split.rc2.val.json', 'r') as f:
            self.name_to_relpath = json.load(f)
    
    def load_model(self, config, model_class):
        """Carica un modello addestrato"""
        model = model_class(config)
        model.load_state_dict(torch.load(config.eval_load_path, map_location='cpu'))
        model.to(self.device)
        model.eval()
        return model
    
    def get_image_path(self, image_name):
        """Restituisce il path completo dell'immagine"""
        rel_path = self.name_to_relpath[image_name][2:]  # Rimuovi './'
        return f"./data/datasets/CIRR/{rel_path}"
    
    def retrieve_images(self, model, reference_name, reference_texts, rel_caption, k=10):
        """Esegue retrieval per una query specifica"""
        with torch.no_grad():
            # Prepara l'input
            reference_texts = [reference_texts]  # Lista di liste di caption
            rel_caption = [rel_caption]
            
            # Converti reference image in tensor
            reference_img_path = self.get_image_path(reference_name)
            reference_img = Image.open(reference_img_path).convert('RGB')
            reference_tensor = self.preprocess(reference_img).unsqueeze(0).to(self.device)
            
            # Combina features
            query_rep = model.combine_features(reference_texts, rel_caption)
            
            # Calcola similarità con tutte le immagini dell'indice
            similarities = (query_rep @ self.index_features.T).squeeze()
            
            # Ordina per similarità
            sorted_indices = torch.argsort(similarities, descending=True)
            retrieved_names = [self.index_names[i] for i in sorted_indices[:k]]
            scores = [similarities[i].item() for i in sorted_indices[:k]]
            
            return retrieved_names, scores
    
    def evaluate_query(self, query_idx, k=10):
        """Valuta una singola query con entrambi i modelli"""
        triplet = self.triplets[query_idx]
        
        reference_name = triplet['reference']
        target_hard = triplet['target_hard']
        rel_caption = triplet['caption']
        reference_texts = [str(x) for x in triplet["multi_caption_opt"]]
        
        print(f"Processing query {query_idx}: {rel_caption}")
        
        # Retrieval con BLIP 1t
        retrieved_1t, scores_1t = self.retrieve_images(
            self.model_1t, reference_name, reference_texts, rel_caption, k
        )
        
        # Retrieval con BLIP 2tv
        retrieved_2tv, scores_2tv = self.retrieve_images(
            self.model_2tv, reference_name, reference_texts, rel_caption, k
        )
        
        return {
            'query_info': triplet,
            'results_1t': {
                'retrieved_images': retrieved_1t,
                'scores': scores_1t
            },
            'results_2tv': {
                'retrieved_images': retrieved_2tv,
                'scores': scores_2tv
            }
        }
    
    def recall_at_k(self, retrieved_list, target, k):
        """Calcola Recall@k"""
        return 1.0 if target in retrieved_list[:k] else 0.0
    
    def find_improvement_cases(self, num_queries=100, top_k=10):
        """Trova casi dove BLIP 2tv performa meglio di BLIP 1t"""
        improvements = []
        
        print(f"Analizzando {num_queries} query per trovare miglioramenti...")
        
        for i in range(min(num_queries, len(self.triplets))):
            try:
                results = self.evaluate_query(i, k=top_k)
                target = results['query_info']['target_hard']
                
                # Calcola Recall@1, @5, @10 per entrambi i modelli
                r1_1t = self.recall_at_k(results['results_1t']['retrieved_images'], target, 1)
                r5_1t = self.recall_at_k(results['results_1t']['retrieved_images'], target, 5)
                r10_1t = self.recall_at_k(results['results_1t']['retrieved_images'], target, 10)
                
                r1_2tv = self.recall_at_k(results['results_2tv']['retrieved_images'], target, 1)
                r5_2tv = self.recall_at_k(results['results_2tv']['retrieved_images'], target, 5)
                r10_2tv = self.recall_at_k(results['results_2tv']['retrieved_images'], target, 10)
                
                # Calcola miglioramento complessivo
                improvement = (r1_2tv + r5_2tv + r10_2tv) - (r1_1t + r5_1t + r10_1t)
                
                if improvement > 0:  # Solo casi con miglioramento
                    improvements.append({
                        'index': i,
                        'r1_1t': r1_1t, 'r5_1t': r5_1t, 'r10_1t': r10_1t,
                        'r1_2tv': r1_2tv, 'r5_2tv': r5_2tv, 'r10_2tv': r10_2tv,
                        'improvement': improvement,
                        'caption': results['query_info']['caption'],
                        'results': results
                    })
                    
                if (i + 1) % 10 == 0:
                    print(f"Processate {i + 1} query, trovati {len(improvements)} miglioramenti")
                    
            except Exception as e:
                print(f"Errore nella query {i}: {e}")
                continue
        
        # Ordina per miglioramento
        improvements.sort(key=lambda x: x['improvement'], reverse=True)
        return improvements

    def add_border_to_image(self, image, color, width=8):
        """Aggiunge un bordo colorato all'immagine"""
        return ImageOps.expand(image, border=width, fill=color)

    def visualize_comparison(self, results, save_path=None):
        query = results['query_info']
        res1 = results['results_1t']
        res2 = results['results_2tv']
        target = query['target_hard']

        # Create compact grid
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(3, 11, figure=fig, hspace=0.05, wspace=0.05) # 3 rows, 11 columns

        # ---- ROW0: Query + Ref + Target ----
        #ax = fig.add_subplot(gs[0, :])
        ax = fig.add_subplot(gs[0, 1:5])
        ax.axis('off')

        wrapped_caption = textwrap.fill(query["caption"], width=50)
        ax.text(0.01, 0.75, f"Query:", fontsize=16, weight='bold', va='center')
        ax.text(0.01, 0.60, f"\"{wrapped_caption}\"", fontsize=15, style='italic', va='center')

        # Reference
        ref_img = Image.open(self.get_image_path(query['reference']))
        ax_ref = fig.add_subplot(gs[0, 6:8]) #7:9
        ax_ref.imshow(ref_img)
        ax_ref.set_title("Reference", fontsize=16, weight='bold')
        ax_ref.axis("off")

        # Target (ground truth)
        tgt_img = Image.open(self.get_image_path(target))
        tgt_img = self.add_border_to_image(tgt_img, 'green', width=8)
        ax_tgt = fig.add_subplot(gs[0, 8:10]) #9:11
        ax_tgt.imshow(tgt_img)
        ax_tgt.set_title("Target", fontsize=16, weight='bold', color='green')
        ax_tgt.axis("off")

        # ---- Function to plot retrieved ----
        def plot_row(imgs, scores, row, title):
            # Label row
            ax_title = fig.add_subplot(gs[row, 0])
            ax_title.text(0.5, 0.5, title, rotation=90,
                        fontsize=16, weight='bold', va='center', ha='center')
            ax_title.axis('off')

            target_width = 100  # Larghezza fissa in pixel per tutte le immagini

            for i, name in enumerate(imgs[:10]):
                ax = fig.add_subplot(gs[row, i+1])
                try:
                    img = Image.open(self.get_image_path(name))

                    # Calcola le nuove dimensioni mantenendo le proporzioni
                    #width_percent = target_height / float(img.size[1])
                    width_percent = target_width / float(img.size[0])
                    #new_width = int(float(img.size[0]) * float(width_percent))
                    new_height = int(float(img.size[1]) * float(width_percent))
                    #img_resized = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
                    img_resized = img.resize((target_width, new_height), Image.Resampling.LANCZOS)

                    is_correct = (name == target)
                    img_resized = self.add_border_to_image(img_resized, 'green' if is_correct else 'red', width=6)
                    #ax.imshow(img)
                    ax.imshow(img_resized)
                    ax.set_title(f"#{i+1}", fontsize=13, color='green' if is_correct else 'black')
                    ax.axis("off")
                except:
                    #print(f"Errore nel caricamento immagine {name}: {e}")
                    ax.text(0.5, 0.5, "ERR", color='r', ha='center')
                    ax.axis("off")

        # ---- ROW1 + ROW2 ----
        plot_row(res1['retrieved_images'], res1['scores'], row=1, title="BLIP-1t")
        plot_row(res2['retrieved_images'], res2['scores'], row=2, title="BLIP-2tv")

        plt.tight_layout(pad=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")

        plt.show()
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Confronto BLIP 1t vs BLIP 2tv su CIRR')
    parser.add_argument('--num_queries', type=int, default=100, help='Numero di query da analizzare')
    parser.add_argument('--top_cases', type=int, default=10, help='Numero di casi migliori da visualizzare')
    parser.add_argument('--output_dir', type=str, default='./comparison_results', help='Directory di output')
    
    args = parser.parse_args()
    
    # Crea directory output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configurazioni per i due modelli
    config_1t = Config()
    config_1t.eval_load_path = "./new/2025-11-03-17-43-13_BLIPtv_cirr_train_50epoch_blipbase_batch16_best_arithmetic"
    config_1t.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config_2tv = Config()
    config_2tv.eval_load_path = "./new/2025-10-21-BLIPtv_cirr_train_50epoch_blipbase_batch16_2textencoder_all_best_arithmetic"
    config_2tv.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Inizializza il comparatore
    print("Inizializzazione CIRR Comparison...")
    comparator = CIRRComparison(config_1t, config_2tv)
    
    # Trova casi con miglioramento
    print(f"\n🔍 Ricerca casi con miglioramento BLIP 2tv...")
    improvements = comparator.find_improvement_cases(
        num_queries=args.num_queries, 
        top_k=10
    )
    
    print(f"\n🎯 Trovati {len(improvements)} casi con miglioramento")
    
    # Visualizza i casi migliori
    for i, case in enumerate(improvements[:args.top_cases]):
        print(f"\n📊 Visualizzazione caso {i+1}:")
        print(f"   Query Index: {case['index']}")
        print(f"   Caption: {case['caption']}")
        print(f"   Miglioramento: {case['improvement']:.3f}")
        print(f"   BLIP 1t - R@1: {case['r1_1t']}, R@5: {case['r5_1t']}, R@10: {case['r10_1t']}")
        print(f"   BLIP 2tv - R@1: {case['r1_2tv']}, R@5: {case['r5_2tv']}, R@10: {case['r10_2tv']}")
        
        # Genera visualizzazione
        save_path = os.path.join(args.output_dir, f'comparison_case_{i+1}_idx{case["index"]}.png')
        comparator.visualize_comparison(case['results'], save_path=save_path)
    
    # Salva riepilogo
    summary_path = os.path.join(args.output_dir, 'improvement_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("RIEPILOGO MIGLIORAMENTI BLIP 2tv vs BLIP 1t\n")
        f.write("=" * 60 + "\n\n")
        
        for i, case in enumerate(improvements[:args.top_cases]):
            f.write(f"Caso {i+1} (Index: {case['index']}):\n")
            f.write(f"Caption: {case['caption']}\n")
            f.write(f"Miglioramento: {case['improvement']:.3f}\n")
            f.write(f"BLIP 1t - R@1: {case['r1_1t']}, R@5: {case['r5_1t']}, R@10: {case['r10_1t']}\n")
            f.write(f"BLIP 2tv - R@1: {case['r1_2tv']}, R@5: {case['r5_2tv']}, R@10: {case['r10_2tv']}\n")
            f.write("-" * 60 + "\n")
    
    print(f"\n💾 Risultati salvati in: {args.output_dir}")
    print(f"📄 Riepilogo: {summary_path}")

if __name__ == '__main__':
    main()

