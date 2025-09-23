# Script per analizzare il risultato del file .json dopo aver applicato la metrica bleu
# Python script that analyzes your JSON file and extracts the top k results for maximum, minimum, and median BLEU scores
import json
from collections import defaultdict
import numpy as np

def analyze_bleu_scores(json_file_path, k=5):
    """
    Analyze BLEU scores from a JSON file and return top k results for max, min, and median scores.
    
    Args:
        json_file_path (str): Path to the JSON file
        k (int): Number of top results to return for each category
    
    Returns:
        dict: Dictionary containing top k results for max, min, and median BLEU scores
    """
    
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Calculate overall median score
    all_scores = [item['bleu_score'] for item in data]
    overall_median = np.median(all_scores)
    
    # Find items closest to the overall median
    median_diff = [abs(item['bleu_score'] - overall_median) for item in data]
    median_indices = np.argsort(median_diff)[:k]
    
    # Get top k for each category
    results = {}
    
    # Top k Max BLEU (individual scores)
    sorted_max = sorted(data, key=lambda x: x['bleu_score'], reverse=True)
    results['top_max_bleu'] = []
    for i, item in enumerate(sorted_max[:k]):
        results['top_max_bleu'].append({
            'ref_image_id': item['ref_image_id'],
            'bleu_score': item['bleu_score'],
            'original_caption': item['original_caption'],
            'generated_caption': item['generated_caption']
        })
    
    # Top k Min BLEU (individual scores)
    sorted_min = sorted(data, key=lambda x: x['bleu_score'])
    results['top_min_bleu'] = []
    for i, item in enumerate(sorted_min[:k]):
        results['top_min_bleu'].append({
            'ref_image_id': item['ref_image_id'],
            'bleu_score': item['bleu_score'],
            'original_caption': item['original_caption'],
            'generated_caption': item['generated_caption']
        })
    
    # Top k Median BLEU (individual scores closest to overall median)
    results['top_median_bleu'] = []
    for i, idx in enumerate(median_indices[:k]):
        item = data[idx]
        results['top_median_bleu'].append({
            'ref_image_id': item['ref_image_id'],
            'bleu_score': item['bleu_score'],
            'original_caption': item['original_caption'],
            'generated_caption': item['generated_caption'],
            'distance_from_median': abs(item['bleu_score'] - overall_median)
        })
    
    # Additional analysis: distribution of scores
    results['overall_stats'] = {
        'mean_score': np.mean(all_scores),
        'median_score': overall_median,
        'std_score': np.std(all_scores),
        'min_score': min(all_scores),
        'max_score': max(all_scores),
        'total_comparisons': len(all_scores),
        'percentile_25': np.percentile(all_scores, 25),
        'percentile_75': np.percentile(all_scores, 75)
    }
    
    # Group data by ref_image_id for additional statistics
    image_data = defaultdict(list)
    for item in data:
        image_id = item['ref_image_id']
        image_data[image_id].append({
            'bleu_score': item['bleu_score'],
            'generated_caption': item['generated_caption']
        })
    
    # Calculate per-image statistics
    image_stats = {}
    for image_id, captions in image_data.items():
        scores = [caption['bleu_score'] for caption in captions]
        image_stats[image_id] = {
            'max_bleu': max(scores),
            'min_bleu': min(scores),
            'median_bleu': np.median(scores),
            'avg_bleu': np.mean(scores),
            'num_captions': len(captions)
        }
    
    results['image_stats'] = {
        'avg_max_per_image': np.mean([stats['max_bleu'] for stats in image_stats.values()]),
        'avg_min_per_image': np.mean([stats['min_bleu'] for stats in image_stats.values()]),
        'avg_median_per_image': np.mean([stats['median_bleu'] for stats in image_stats.values()]),
        'total_images': len(image_stats)
    }
    
    return results

def print_results(results, k):
    """Print the analysis results in a readable format."""
    
    print(f"=== OVERALL STATISTICS ===")
    stats = results['overall_stats']
    print(f"Mean BLEU Score: {stats['mean_score']:.6f}")
    print(f"Median BLEU Score: {stats['median_score']:.6f}")
    print(f"Standard Deviation: {stats['std_score']:.6f}")
    print(f"Min Score: {stats['min_score']:.6f}")
    print(f"Max Score: {stats['max_score']:.6f}")
    print(f"25th Percentile: {stats['percentile_25']:.6f}")
    print(f"75th Percentile: {stats['percentile_75']:.6f}")
    print(f"Total Comparisons: {stats['total_comparisons']}")
    
    img_stats = results['image_stats']
    print(f"\n=== PER IMAGE STATISTICS ===")
    print(f"Total Images: {img_stats['total_images']}")
    print(f"Average Max Score per Image: {img_stats['avg_max_per_image']:.6f}")
    print(f"Average Min Score per Image: {img_stats['avg_min_per_image']:.6f}")
    print(f"Average Median Score per Image: {img_stats['avg_median_per_image']:.6f}")
    
    print(f"\n=== TOP {k} MAX BLEU SCORES (Migliori score individuali) ===")
    for i, item in enumerate(results['top_max_bleu'], 1):
        print(f"\n{i}. Image ID: {item['ref_image_id']}")
        print(f"   BLEU Score: {item['bleu_score']:.6f}")
        print(f"   Original: {item['original_caption']}")
        print(f"   Generated: {item['generated_caption']}")
    
    print(f"\n=== TOP {k} MIN BLEU SCORES (Peggiori score individuali) ===")
    for i, item in enumerate(results['top_min_bleu'], 1):
        print(f"\n{i}. Image ID: {item['ref_image_id']}")
        print(f"   BLEU Score: {item['bleu_score']:.6f}")
        print(f"   Original: {item['original_caption']}")
        print(f"   Generated: {item['generated_caption']}")
    
    print(f"\n=== TOP {k} CLOSEST TO MEDIAN BLEU SCORES (Score più vicini alla mediana globale) ===")
    print(f"Mediana globale: {results['overall_stats']['median_score']:.6f}")
    for i, item in enumerate(results['top_median_bleu'], 1):
        print(f"\n{i}. Image ID: {item['ref_image_id']}")
        print(f"   BLEU Score: {item['bleu_score']:.6f}")
        print(f"   Distanza dalla mediana: {item['distance_from_median']:.8f}")
        print(f"   Original: {item['original_caption']}")
        print(f"   Generated: {item['generated_caption']}")

# Example usage
if __name__ == "__main__":
    # Replace with your JSON file path
    json_file_path = "./data/files/bleu_scores_sdam_auto_fixed.json"
    k = 5  # Number of top results to show
    
    try:
        results = analyze_bleu_scores(json_file_path, k)
        print_results(results, k)
        
        # Optional: Save results to a new JSON file
        with open("./data/files/bleu_analysis_results_individual.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to 'bleu_analysis_results_individual.json'")
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file_path}'.")
    except Exception as e:
        print(f"Error: {e}")