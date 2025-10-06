#!/usr/bin/env python3
"""
Compare token counts and pricing across different OpenAI models for AgentNet data.
"""

import json
import argparse
from typing import Dict, Any, List
from tabulate import tabulate


def load_results(file_path: str) -> Dict[str, Any]:
    """Load token calculation results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def format_number(num):
    """Format number with commas."""
    if isinstance(num, int):
        return f"{num:,}"
    elif isinstance(num, float):
        return f"{num:.2f}"
    return str(num)


def compare_models(result_files: Dict[str, str]):
    """Compare token counts across different models."""
    results = {}
    
    # Load all result files
    for model_name, file_path in result_files.items():
        try:
            results[model_name] = load_results(file_path)
            print(f"Loaded results for {model_name} from {file_path}")
        except FileNotFoundError:
            print(f"Warning: Could not find results file for {model_name}: {file_path}")
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not results:
        print("No valid result files found!")
        return
    
    models = list(results.keys())
    
    # Prepare comparison table
    print("\n" + "=" * 100)
    print("MODEL COMPARISON - TOKEN COUNTS")
    print("=" * 100)
    
    # Token metrics table
    token_headers = ["Metric"] + models
    token_rows = []
    
    # Add rows
    token_rows.append(["Conversations"] + [format_number(results[m]['total_conversations']) for m in models])
    token_rows.append(["Total Text Tokens"] + [format_number(results[m]['totals']['total_text_tokens']) for m in models])
    token_rows.append(["Total Image Tokens"] + [format_number(results[m]['totals']['total_image_tokens']) for m in models])
    token_rows.append(["Total Input Tokens"] + [format_number(results[m]['totals']['total_input_tokens']) for m in models])
    token_rows.append(["Total Output Tokens"] + [format_number(results[m]['totals']['total_output_tokens']) for m in models])
    token_rows.append(["Total Tokens"] + [format_number(results[m]['totals']['total_tokens']) for m in models])
    token_rows.append(["Total Images"] + [format_number(results[m]['totals']['total_images']) for m in models])
    
    print(tabulate(token_rows, headers=token_headers, tablefmt="grid"))
    
    # Averages table
    print("\n" + "=" * 100)
    print("MODEL COMPARISON - AVERAGES PER CONVERSATION")
    print("=" * 100)
    
    avg_headers = ["Metric"] + models
    avg_rows = []
    
    avg_rows.append(["Avg Text Tokens"] + [f"{results[m]['averages']['avg_text_tokens']:.1f}" for m in models])
    avg_rows.append(["Avg Image Tokens"] + [f"{results[m]['averages']['avg_image_tokens']:.1f}" for m in models])
    avg_rows.append(["Avg Input Tokens"] + [f"{results[m]['averages']['avg_total_input_tokens']:.1f}" for m in models])
    avg_rows.append(["Avg Output Tokens"] + [f"{results[m]['averages']['avg_output_tokens']:.1f}" for m in models])
    avg_rows.append(["Avg Total Tokens"] + [f"{results[m]['averages']['avg_total_tokens']:.1f}" for m in models])
    avg_rows.append(["Avg Images"] + [f"{results[m]['averages']['avg_images_per_conversation']:.1f}" for m in models])
    
    print(tabulate(avg_rows, headers=avg_headers, tablefmt="grid"))
    
    # Pricing comparison
    print("\n" + "=" * 100)
    print("MODEL COMPARISON - PRICING (based on official OpenAI rates)")
    print("=" * 100)
    
    pricing_headers = ["Model", "Input Rate", "Output Rate", "Input Cost", "Output Cost", "Total Cost", "Cost/Conv"]
    pricing_rows = []
    
    for model in models:
        config = results[model]['model_config']
        totals = results[model]['totals']
        averages = results[model]['averages']
        
        row = [
            model,
            f"${config['input_price_per_1m']:.2f}/1M",
            f"${config['output_price_per_1m']:.2f}/1M",
            f"${totals['total_input_cost']:.2f}",
            f"${totals['total_output_cost']:.2f}",
            f"${totals['total_cost']:.2f}",
            f"${averages['avg_cost_per_conversation']:.4f}"
        ]
        pricing_rows.append(row)
    
    print(tabulate(pricing_rows, headers=pricing_headers, tablefmt="grid"))
    
    # Model features comparison
    print("\n" + "=" * 100)
    print("MODEL FEATURES")
    print("=" * 100)
    
    features_headers = ["Model", "Encoding", "Vision Support", "Input $/1M", "Output $/1M"]
    features_rows = []
    
    for model in models:
        config = results[model]['model_config']
        row = [
            model,
            config['encoding'],
            "Yes" if config['supports_vision'] else "No",
            f"${config['input_price_per_1m']:.2f}",
            f"${config['output_price_per_1m']:.2f}"
        ]
        features_rows.append(row)
    
    print(tabulate(features_rows, headers=features_headers, tablefmt="grid"))
    
    # Cost efficiency analysis
    print("\n" + "=" * 100)
    print("COST EFFICIENCY ANALYSIS")
    print("=" * 100)
    
    # Rank by total cost
    model_costs = [(m, results[m]['totals']['total_cost']) for m in models]
    model_costs.sort(key=lambda x: x[1])
    
    print("\nModels ranked by total cost (lowest to highest):")
    for i, (model, cost) in enumerate(model_costs, 1):
        conversations = results[model]['total_conversations']
        cost_per_conv = results[model]['averages']['avg_cost_per_conversation']
        print(f"  {i}. {model:20s} - Total: ${cost:8.2f} | Per conv: ${cost_per_conv:.4f} | Convs: {conversations:,}")
    
    # Calculate cost multipliers relative to cheapest
    if len(model_costs) > 1:
        cheapest_cost = model_costs[0][1]
        cheapest_convs = results[model_costs[0][0]]['total_conversations']
        
        print(f"\nCost multipliers relative to {model_costs[0][0]} (normalized per conversation):")
        for model, cost in model_costs:
            convs = results[model]['total_conversations']
            normalized_cost = (cost / convs) if convs > 0 else 0
            cheapest_normalized = (cheapest_cost / cheapest_convs) if cheapest_convs > 0 else 1
            multiplier = normalized_cost / cheapest_normalized if cheapest_normalized > 0 else 0
            print(f"  {model:20s} - {multiplier:.2f}x")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare token counts across OpenAI models')
    parser.add_argument('--results', '-r', 
                       action='append',
                       nargs=2,
                       metavar=('MODEL', 'FILE'),
                       help='Model name and results file (can be used multiple times)')
    parser.add_argument('--all', '-a',
                       action='store_true',
                       help='Compare all available result files in current directory')
    
    args = parser.parse_args()
    
    result_files = {}
    
    if args.all:
        # Auto-discover result files
        import glob
        import os
        
        for file in glob.glob("*_tokens*.json"):
            # Extract model name from filename
            model_name = os.path.basename(file).replace('_tokens.json', '').replace('_tokens_accurate.json', '')
            result_files[model_name] = file
        
        if not result_files:
            print("No result files found matching pattern '*_tokens*.json'")
            return
        
        print(f"Auto-discovered {len(result_files)} result files")
    elif args.results:
        # Use specified files
        result_files = {model: file for model, file in args.results}
    else:
        # Default comparison
        print("No files specified. Use --results or --all flag.")
        print("\nExample usage:")
        print("  python compare_models_pyspark.py -r gpt-4o gpt4o_tokens_accurate.json -r o1-mini o1mini_tokens.json")
        print("  python compare_models_pyspark.py --all")
        return
    
    compare_models(result_files)


if __name__ == "__main__":
    main()

