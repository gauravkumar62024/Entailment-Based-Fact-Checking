import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt


def save_length_histogram(lengths, output_path, model_name, bins=100):
    """
    Save histogram image of sequence length distribution
    """
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=bins)
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Number of Examples")
    plt.title(f"Sequence Length Distribution\nTokenizer: {model_name}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_length_histogram_log(lengths, output_path, model_name, bins=100):
    """
    Save log-scaled histogram image
    """
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=bins, log=True)
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Number of Examples (log scale)")
    plt.title(f"Sequence Length Distribution (Log Scale)\nTokenizer: {model_name}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_sequence_lengths(file_path, model_name):
    """
    Analyze sequence lengths of instruction + input in the converted dataset
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the converted dataset
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sequence_lengths = []
    
    for example in data:
        # Combine instruction and input
        full_text = example["instruction"] + "\n\n" + example["input"]
        
        # Tokenize and count tokens
        tokens = tokenizer.encode(full_text)
        sequence_lengths.append(len(tokens))
    
    # Calculate statistics
    if sequence_lengths:
        stats = {
            "file": str(file_path),
            "model_tokenizer": model_name,
            "total_examples": len(sequence_lengths),
            "avg_length": float(np.mean(sequence_lengths)),
            "max_length": int(np.max(sequence_lengths)),
            "min_length": int(np.min(sequence_lengths)),
            "std_length": float(np.std(sequence_lengths)),
            "median_length": float(np.median(sequence_lengths)),
            "p95_length": float(np.percentile(sequence_lengths, 95)),
            "p99_length": float(np.percentile(sequence_lengths, 99))
        }
    else:
        stats = {
            "file": str(file_path),
            "model_tokenizer": model_name,
            "total_examples": 0,
            "avg_length": 0,
            "max_length": 0,
            "min_length": 0,
            "std_length": 0,
            "median_length": 0,
            "p95_length": 0,
            "p99_length": 0
        }
    
    return stats, sequence_lengths

def main():
    parser = argparse.ArgumentParser(description="Analyze sequence lengths in converted dataset files")
    parser.add_argument("--input_file", required=True, help="Path to converted JSON file")
    parser.add_argument("--model_name", required=True, help="Model name for tokenizer (e.g., 'meta-llama/Llama-2-7b-hf', 'microsoft/deberta-v3-base')")
    parser.add_argument("--output_dir", default=".", help="Output directory for statistics files (default: current directory)")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return
    
    print(f"Analyzing sequence lengths for: {input_file}")
    print(f"Using tokenizer from: {args.model_name}")
    print("-" * 60)
    
    try:
        stats, all_lengths = analyze_sequence_lengths(input_file, args.model_name)
        
        # Print statistics
        print(f"File: {stats['file']}")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Average length: {stats['avg_length']:.2f} tokens")
        print(f"Max length: {stats['max_length']} tokens")
        print(f"Min length: {stats['min_length']} tokens")
        print(f"Std length: {stats['std_length']:.2f} tokens")
        print(f"Median length: {stats['median_length']:.2f} tokens")
        print(f"95th percentile: {stats['p95_length']:.2f} tokens")
        print(f"99th percentile: {stats['p99_length']:.2f} tokens")
        print("-" * 60)
        
        # Print histogram-like summary
        if all_lengths:
            print("Length distribution summary:")
            bins = [0, 256, 512, 1024, 2048, 4096, 8192, float('inf')]
            labels = ["<256", "256-512", "512-1024", "1024-2048", "2048-4096", "4096-8192", ">8192"]
            
            counts = []
            for i in range(len(bins)-1):
                if i == len(bins)-2:  # Last bin
                    count = sum(1 for length in all_lengths if length >= bins[i])
                else:
                    count = sum(1 for length in all_lengths if bins[i] <= length < bins[i+1])
                counts.append(count)
                percentage = (count / len(all_lengths)) * 100
                print(f"  {labels[i]:<10}: {count:>6} examples ({percentage:>6.2f}%)")
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename with "statistic_of_" prefix
        output_filename = f"statistic_of_{input_file.name}"
        stats_file = output_dir / output_filename
        # Save histogram image
        histogram_file = output_dir / f"histogram_of_{input_file.stem}.png"
        save_length_histogram(
            all_lengths,
            histogram_file,
            args.model_name
        )
        print(f"Histogram saved to: {histogram_file}")

        # Save log-scale histogram
        histogram_log_file = output_dir / f"histogram_log_of_{input_file.stem}.png"
        save_length_histogram_log(
            all_lengths,
            histogram_log_file,
            args.model_name
        )
        print(f"Log-scale histogram saved to: {histogram_log_file}")

        # Save statistics to file
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to: {stats_file}")
        
        # Also save the full length distribution with a different prefix
        distribution_filename = f"distribution_of_{input_file.name}"
        distribution_file = output_dir / distribution_filename
        distribution_data = {
            "lengths": all_lengths,
            "stats": stats
        }
        with open(distribution_file, 'w', encoding='utf-8') as f:
            json.dump(distribution_data, f, indent=2, ensure_ascii=False)
        print(f"Full distribution saved to: {distribution_file}")
        
        # Save a human-readable summary
        summary_filename = f"summary_of_{input_file.name.replace('.json', '.txt')}"
        summary_file = output_dir / summary_filename
        model_name = args.model_name
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Sequence Length Analysis\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Tokenizer: {model_name}\n")
            f.write(f"Total examples: {stats['total_examples']}\n\n")
            
            f.write(f"Statistics:\n")
            f.write(f"  Average length: {stats['avg_length']:.2f} tokens\n")
            f.write(f"  Max length: {stats['max_length']} tokens\n")
            f.write(f"  Min length: {stats['min_length']} tokens\n")
            f.write(f"  Std length: {stats['std_length']:.2f} tokens\n")
            f.write(f"  Median length: {stats['median_length']:.2f} tokens\n")
            f.write(f"  95th percentile: {stats['p95_length']:.2f} tokens\n")
            f.write(f"  99th percentile: {stats['p99_length']:.2f} tokens\n\n")
            
            f.write(f"Length Distribution:\n")
            for i in range(len(bins)-1):
                count = counts[i]
                percentage = (count / len(all_lengths)) * 100
                f.write(f"  {labels[i]:<10}: {count:>6} examples ({percentage:>6.2f}%)\n")
        
        print(f"Human-readable summary saved to: {summary_file}")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()