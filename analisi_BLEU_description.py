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
    
    # Group data by ref_image_id
    image_data = defaultdict(list)
    for item in data:
        image_id = item['ref_image_id']
        image_data[image_id].append({
            'bleu_score': item['bleu_score'],
            'original_caption': item['original_caption'],
            'generated_caption': item['generated_caption']
        })
    
    # Calculate statistics for each image
    image_stats = {}
    for image_id, captions in image_data.items():
        scores = [caption['bleu_score'] for caption in captions]
        image_stats[image_id] = {
            'max_bleu': max(scores),
            'min_bleu': min(scores),
            'median_bleu': np.median(scores),
            'avg_bleu': sum(scores) / len(scores),
            'num_captions': len(captions),
            'captions': captions
        }
    
    # Get top k for each category
    results = {}
    
    # Top k Max BLEU
    sorted_max = sorted(image_stats.items(), key=lambda x: x[1]['max_bleu'], reverse=True)
    results['top_max_bleu'] = []
    for i, (image_id, stats) in enumerate(sorted_max[:k]):
        # Find the caption with max BLEU score
        max_caption = max(stats['captions'], key=lambda x: x['bleu_score'])
        results['top_max_bleu'].append({
            'ref_image_id': image_id,
            'bleu_score': stats['max_bleu'],
            'original_caption': max_caption['original_caption'],
            'generated_caption': max_caption['generated_caption'],
            'median_bleu': stats['median_bleu'],
            'avg_bleu': stats['avg_bleu']
        })
    
    # Top k Min BLEU
    sorted_min = sorted(image_stats.items(), key=lambda x: x[1]['min_bleu'])
    results['top_min_bleu'] = []
    for i, (image_id, stats) in enumerate(sorted_min[:k]):
        # Find the caption with min BLEU score
        min_caption = min(stats['captions'], key=lambda x: x['bleu_score'])
        results['top_min_bleu'].append({
            'ref_image_id': image_id,
            'bleu_score': stats['min_bleu'],
            'original_caption': min_caption['original_caption'],
            'generated_caption': min_caption['generated_caption'],
            'median_bleu': stats['median_bleu'],
            'avg_bleu': stats['avg_bleu']
        })
    
    # Top k Median BLEU
    sorted_median = sorted(image_stats.items(), key=lambda x: x[1]['median_bleu'], reverse=True)
    results['top_median_bleu'] = []
    for i, (image_id, stats) in enumerate(sorted_median[:k]):
        # Find a caption closest to the median for representation
        median_caption = min(stats['captions'], key=lambda x: abs(x['bleu_score'] - stats['median_bleu']))
        results['top_median_bleu'].append({
            'ref_image_id': image_id,
            'bleu_score': stats['median_bleu'],
            'original_caption': stats['captions'][0]['original_caption'],  # All should be same
            'generated_caption': median_caption['generated_caption'],
            'avg_bleu': stats['avg_bleu'],
            'num_captions': stats['num_captions'],
            'all_captions': stats['captions']  # Include all for reference
        })
    
    return results

def print_results(results, k):
    """Print the analysis results in a readable format."""
    
    print(f"=== TOP {k} MAX BLEU SCORES ===")
    for i, item in enumerate(results['top_max_bleu'], 1):
        print(f"\n{i}. Image ID: {item['ref_image_id']}")
        print(f"   Max BLEU Score: {item['bleu_score']:.4f}")
        print(f"   Median BLEU: {item['median_bleu']:.4f}")
        print(f"   Average BLEU: {item['avg_bleu']:.4f}")
        print(f"   Original: {item['original_caption']}")
        print(f"   Generated: {item['generated_caption']}")
    
    print(f"\n=== TOP {k} MIN BLEU SCORES ===")
    for i, item in enumerate(results['top_min_bleu'], 1):
        print(f"\n{i}. Image ID: {item['ref_image_id']}")
        print(f"   Min BLEU Score: {item['bleu_score']:.4f}")
        print(f"   Median BLEU: {item['median_bleu']:.4f}")
        print(f"   Average BLEU: {item['avg_bleu']:.4f}")
        print(f"   Original: {item['original_caption']}")
        print(f"   Generated: {item['generated_caption']}")
    
    print(f"\n=== TOP {k} MEDIAN BLEU SCORES ===")
    for i, item in enumerate(results['top_median_bleu'], 1):
        print(f"\n{i}. Image ID: {item['ref_image_id']}")
        print(f"   Median BLEU: {item['bleu_score']:.4f}")
        print(f"   Average BLEU: {item['avg_bleu']:.4f}")
        print(f"   Number of captions: {item['num_captions']}")
        print(f"   Original: {item['original_caption']}")
        print(f"   Generated (closest to median): {item['generated_caption']}")

# Example usage
if __name__ == "__main__":
    # Replace with your JSON file path
    json_file_path = "./data/files/bleu_scores_sdam_auto_fixed.json"
    k = 5  # Number of top results to show
    
    try:
        results = analyze_bleu_scores(json_file_path, k)
        print_results(results, k)
        
        # Optional: Save results to a new JSON file
        with open("./data/files/bleu_analysis_results2.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to 'bleu_analysis_results2.json'")
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file_path}'.")
    except Exception as e:
        print(f"Error: {e}")