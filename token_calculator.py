#!/usr/bin/env python3
"""
PySpark script to calculate OpenAI token counts for AgentNet Ubuntu training data.

This script accurately calculates:
- Text tokens using OpenAI's tiktoken library
- Image tokens based on actual image dimensions and OpenAI's tiling algorithm
- Pricing based on official OpenAI pricing documentation
"""

import argparse
import json
import os
from typing import Dict, List, Any, Tuple
from PIL import Image
import math

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, LongType, FloatType

try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken not installed. Run: pip install tiktoken")
    exit(1)

# OpenAI model configurations based on official documentation
# Updated as of December 2024
# Image token multipliers from: https://platform.openai.com/docs/guides/images-vision#calculating-costs
OPENAI_MODELS = {
    # GPT-4o models
    'gpt-4o': {
        'encoding': 'o200k_base',
        'supports_vision': True,
        'input_price_per_1m': 2.50,   # $2.50 per 1M input tokens
        'output_price_per_1m': 10.00,  # $10.00 per 1M output tokens
        'image_token_multiplier': 1.0,  # No multiplier for gpt-4o
    },
    'gpt-4o-mini': {
        'encoding': 'o200k_base',
        'supports_vision': True,
        'input_price_per_1m': 0.15,   # $0.15 per 1M input tokens
        'output_price_per_1m': 0.60,  # $0.60 per 1M output tokens
        'image_token_multiplier': 33.33,  # gpt-4o-mini uses 33.33x multiplier for image tokens
    },
    'gpt-4o-2024-11-20': {
        'encoding': 'o200k_base',
        'supports_vision': True,
        'input_price_per_1m': 2.50,
        'output_price_per_1m': 10.00,
        'image_token_multiplier': 1.0,  # No multiplier for gpt-4o
    },
    'gpt-4o-2024-08-06': {
        'encoding': 'o200k_base',
        'supports_vision': True,
        'input_price_per_1m': 2.50,
        'output_price_per_1m': 10.00,
        'image_token_multiplier': 1.0,  # No multiplier for gpt-4o
    },

    # Later oN-mini models (reasoning models)
    'o4-mini': {
        'encoding': 'o200k_base',
        'supports_vision': True,
        'input_price_per_1m': 1.10,
        'output_price_per_1m': 0.275,
        'image_token_multiplier': 1.72, 
    },

    # GPT-5 Models
    'gpt-5': {
        'encoding': 'o200k_harmony',
        'supports_vision': True,
        'input_price_per_1m': 1.25,
        'output_price_per_1m': 10.00,
        'image_token_multiplier': 1.0,  # Assuming no multiplier (update if documented)
    },
    'gpt-5-mini': {
        'encoding': 'o200k_harmony',
        'supports_vision': True,
        'input_price_per_1m': 0.25,
        'output_price_per_1m': 2.00,
        'image_token_multiplier': 1.62,  # Assuming no multiplier (update if documented)
    },
}

# Fixed output tokens per conversation
FIXED_OUTPUT_TOKENS = 8092


class TokenCalculator:
    """Calculate token counts for OpenAI models with accurate image token calculation."""
    
    def __init__(self, model_name: str = 'gpt-4o'):
        """Initialize tokenizer for specified model."""
        self.model_name = model_name
        self.model_config = OPENAI_MODELS.get(model_name)
        
        if not self.model_config:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(OPENAI_MODELS.keys())}"
            )
        
        try:
            self.tokenizer = tiktoken.get_encoding(self.model_config['encoding'])
        except Exception as e:
            raise RuntimeError(f"Could not load tokenizer for {model_name}: {e}")
    
    def count_text_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        if not text:
            return 0
        
        return len(self.tokenizer.encode(text))
    
    def calculate_image_tokens(self, image_path: str) -> int:
        """
        Calculate image tokens based on OpenAI's official algorithm.
        
        From OpenAI documentation:
        1. Images are first scaled to fit within 2048 x 2048 square, maintaining aspect ratio
        2. Then scaled such that shortest side is 768px
        3. Image is divided into 512px tiles
        4. Each tile costs 170 tokens
        5. Base cost of 85 tokens per image
        6. Apply model-specific image token multiplier (e.g., 33.33x for gpt-4o-mini)
        
        Reference: https://platform.openai.com/docs/guides/vision
        Reference: https://platform.openai.com/docs/guides/images-vision#calculating-costs
        """
        if not self.model_config['supports_vision']:
            return 0
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return 0
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Step 1: Scale to fit within 2048x2048
            max_dimension = 2048
            if width > max_dimension or height > max_dimension:
                if width > height:
                    height = int(height * max_dimension / width)
                    width = max_dimension
                else:
                    width = int(width * max_dimension / height)
                    height = max_dimension
            
            # Step 2: Scale so shortest side is 768px
            min_dimension = 768
            if width < height:
                if width > min_dimension:
                    height = int(height * min_dimension / width)
                    width = min_dimension
            else:
                if height > min_dimension:
                    width = int(width * min_dimension / height)
                    height = min_dimension
            
            # Step 3: Calculate number of 512px tiles
            tile_size = 512
            num_tiles_width = math.ceil(width / tile_size)
            num_tiles_height = math.ceil(height / tile_size)
            num_tiles = num_tiles_width * num_tiles_height
            
            # Step 4: Calculate base tokens
            # 85 base tokens + 170 tokens per tile
            base_image_tokens = 85 + (170 * num_tiles)
            
            # Step 5: Apply model-specific image token multiplier
            # Some models like gpt-4o-mini apply a 33.33x multiplier for billing
            multiplier = self.model_config.get('image_token_multiplier', 1.0)
            image_tokens = int(base_image_tokens * multiplier)
            
            return image_tokens
            
        except Exception as e:
            print(f"Warning: Error processing image {image_path}: {e}")
            # Return a conservative estimate if image can't be processed
            multiplier = self.model_config.get('image_token_multiplier', 1.0)
            return int((85 + 170) * multiplier)  # Minimum: 1 tile with multiplier
    
    def calculate_pricing(self, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Calculate pricing based on official OpenAI pricing."""
        input_cost = (input_tokens / 1_000_000) * self.model_config['input_price_per_1m']
        output_cost = (output_tokens / 1_000_000) * self.model_config['output_price_per_1m']
        total_cost = input_cost + output_cost
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        }


def calculate_conversation_tokens(conversation_data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Calculate token counts for a single conversation."""
    calculator = TokenCalculator(model_name)
    
    text_tokens = 0
    image_tokens = 0
    image_details = []
    
    # Count text tokens from messages
    messages = conversation_data.get('messages', [])
    for message in messages:
        content = message.get('content', '')
        if content != '<image>':
            # This is text content - count tokens accurately
            text_tokens += calculator.count_text_tokens(content)
    
    # Calculate image tokens from actual image files
    images = conversation_data.get('images', [])
    multiplier = calculator.model_config.get('image_token_multiplier', 1.0)
    
    for image_path in images:
        img_tokens = calculator.calculate_image_tokens(image_path)
        image_tokens += img_tokens
        
        # Store both multiplied and base tokens for transparency
        base_tokens = int(img_tokens / multiplier) if multiplier > 1.0 else img_tokens
        image_details.append({
            'path': image_path,
            'tokens': img_tokens,
            'base_tokens': base_tokens,
            'multiplier': multiplier
        })
    
    total_input_tokens = text_tokens + image_tokens
    
    # Calculate pricing
    pricing = calculator.calculate_pricing(total_input_tokens, FIXED_OUTPUT_TOKENS)
    
    return {
        'text_tokens': text_tokens,
        'image_tokens': image_tokens,
        'image_count': len(images),
        'total_input_tokens': total_input_tokens,
        'output_tokens': FIXED_OUTPUT_TOKENS,
        'total_tokens': total_input_tokens + FIXED_OUTPUT_TOKENS,
        'image_details': image_details,
        'pricing': pricing
    }


def main():
    """Main function to run the PySpark token calculation job."""
    parser = argparse.ArgumentParser(description='Calculate OpenAI token counts for AgentNet data')
    parser.add_argument('--input-file', '-i', 
                       default='agentnet_ubuntu_5k_train.json',
                       help='Input JSON file path')
    parser.add_argument('--model', '-m',
                       default='gpt-5',
                       choices=list(OPENAI_MODELS.keys()),
                       help='OpenAI model for tokenization')
    parser.add_argument('--output-file', '-o',
                       default='token_counts_detailed.json',
                       help='Output file for token statistics')
    parser.add_argument('--show-sample', '-s',
                       action='store_true',
                       help='Show sample calculations for first few entries')
    parser.add_argument('--limit', '-l',
                       type=int,
                       help='Limit processing to first N conversations (for testing)')
    
    args = parser.parse_args()
    
    # Initialize Spark session with configuration similar to existing scripts
    conf = SparkConf() \
        .setAppName("AgentNet Token Calculator") \
        .setMaster("local[*]") \
        .set("spark.driver.memory", "32g") \
        .set("spark.executor.memory", "16g") \
        .set("spark.driver.maxResultSize", "8g") \
        .set("spark.sql.adaptive.enabled", "true") \
        .set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    
    try:
        print(f"{'='*80}")
        print(f"AgentNet Token Calculator - PySpark Version")
        print(f"{'='*80}")
        print(f"Model: {args.model}")
        print(f"Spark driver memory: {sc.getConf().get('spark.driver.memory')}")
        print(f"Spark executor memory: {sc.getConf().get('spark.executor.memory')}")
        
        # Check if input file exists
        if not os.path.exists(args.input_file):
            # Try alternative paths
            alt_paths = [
                f"/local/scratch/lin.3976/AgentNet/{args.input_file}",
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    args.input_file = alt_path
                    break
            else:
                raise FileNotFoundError(f"Could not find input file: {args.input_file}")
        
        print(f"Reading data from: {args.input_file}")
        
        # Read JSON data
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} conversations")
        
        # Apply limit if specified
        if args.limit:
            data = data[:args.limit]
            print(f"Processing limited to first {len(data)} conversations")
        
        # Create RDD and process in parallel using PySpark
        print("Processing conversations with PySpark...")
        data_rdd = sc.parallelize(data)
        
        # Map function to calculate tokens for each conversation
        def process_conversation(conv_with_index):
            idx, conv = conv_with_index
            try:
                result = calculate_conversation_tokens(conv, args.model)
                result['conversation_id'] = idx
                return result
            except Exception as e:
                print(f"Error processing conversation {idx}: {e}")
                return None
        
        # Add index and process
        indexed_data = data_rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
        results_rdd = indexed_data.map(process_conversation).filter(lambda x: x is not None)
        
        # Collect results
        results = results_rdd.collect()
        
        print(f"Successfully processed {len(results)} conversations")
        
        # Show sample calculations
        if args.show_sample:
            print(f"\n{'='*80}")
            print("SAMPLE CALCULATIONS")
            print(f"{'='*80}")
            for i, result in enumerate(results[:3]):
                conv_id = result['conversation_id']
                print(f"\nConversation {conv_id}:")
                print(f"  Text tokens: {result['text_tokens']:,}")
                print(f"  Image tokens: {result['image_tokens']:,}")
                print(f"  Images: {result['image_count']}")
                print(f"  Input tokens: {result['total_input_tokens']:,}")
                print(f"  Output tokens: {result['output_tokens']:,}")
                print(f"  Total tokens: {result['total_tokens']:,}")
                print(f"  Cost: ${result['pricing']['total_cost']:.4f}")
                if result['image_details']:
                    print(f"  Image details:")
                    for img_detail in result['image_details'][:2]:  # Show first 2 images
                        multiplier = img_detail.get('multiplier', 1.0)
                        if multiplier > 1.0:
                            base = img_detail.get('base_tokens', img_detail['tokens'])
                            print(f"    - {os.path.basename(img_detail['path'])}: {img_detail['tokens']:,} tokens (base: {base}, multiplier: {multiplier}x)")
                        else:
                            print(f"    - {os.path.basename(img_detail['path'])}: {img_detail['tokens']:,} tokens")
        
        # Calculate aggregate statistics using reduce
        total_text_tokens = sum(r['text_tokens'] for r in results)
        total_image_tokens = sum(r['image_tokens'] for r in results)
        total_input_tokens = sum(r['total_input_tokens'] for r in results)
        total_output_tokens = sum(r['output_tokens'] for r in results)
        total_tokens = sum(r['total_tokens'] for r in results)
        total_images = sum(r['image_count'] for r in results)
        total_cost = sum(r['pricing']['total_cost'] for r in results)
        total_input_cost = sum(r['pricing']['input_cost'] for r in results)
        total_output_cost = sum(r['pricing']['output_cost'] for r in results)
        
        # Calculate averages
        count = len(results)
        avg_text_tokens = total_text_tokens / count
        avg_image_tokens = total_image_tokens / count
        avg_total_input_tokens = total_input_tokens / count
        avg_output_tokens = total_output_tokens / count
        avg_total_tokens = total_tokens / count
        avg_images_per_conversation = total_images / count
        avg_cost_per_conversation = total_cost / count
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"TOKEN CALCULATION SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {args.model}")
        print(f"Conversations processed: {count:,}")
        print(f"\nTOKEN TOTALS:")
        print(f"  Text tokens: {total_text_tokens:,}")
        print(f"  Image tokens: {total_image_tokens:,}")
        print(f"  Total input tokens: {total_input_tokens:,}")
        print(f"  Total output tokens: {total_output_tokens:,}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Total images: {total_images:,}")
        
        print(f"\nAVERAGES PER CONVERSATION:")
        print(f"  Avg text tokens: {avg_text_tokens:.1f}")
        print(f"  Avg image tokens: {avg_image_tokens:.1f}")
        print(f"  Avg total input tokens: {avg_total_input_tokens:.1f}")
        print(f"  Avg output tokens: {avg_output_tokens:.1f}")
        print(f"  Avg total tokens: {avg_total_tokens:.1f}")
        print(f"  Avg images: {avg_images_per_conversation:.1f}")
        
        print(f"\nPRICING (based on official OpenAI rates):")
        print(f"  Input cost: ${total_input_cost:.2f}")
        print(f"  Output cost: ${total_output_cost:.2f}")
        print(f"  Total cost: ${total_cost:.2f}")
        print(f"  Avg cost per conversation: ${avg_cost_per_conversation:.4f}")
        
        model_config = OPENAI_MODELS[args.model]
        print(f"\nPRICING DETAILS:")
        print(f"  Input: ${model_config['input_price_per_1m']:.2f} per 1M tokens")
        print(f"  Output: ${model_config['output_price_per_1m']:.2f} per 1M tokens")
        print(f"  Vision support: {model_config['supports_vision']}")
        if model_config['supports_vision']:
            multiplier = model_config.get('image_token_multiplier', 1.0)
            print(f"  Image token multiplier: {multiplier}x")
            if multiplier > 1.0:
                print(f"  Note: Image tokens are multiplied by {multiplier} for billing")
        
        # Save detailed results
        output_data = {
            'model': args.model,
            'model_config': {
                'encoding': model_config['encoding'],
                'supports_vision': model_config['supports_vision'],
                'input_price_per_1m': model_config['input_price_per_1m'],
                'output_price_per_1m': model_config['output_price_per_1m'],
                'image_token_multiplier': model_config.get('image_token_multiplier', 1.0),
            },
            'input_file': args.input_file,
            'total_conversations': count,
            'totals': {
                'total_text_tokens': total_text_tokens,
                'total_image_tokens': total_image_tokens,
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'total_tokens': total_tokens,
                'total_images': total_images,
                'total_input_cost': total_input_cost,
                'total_output_cost': total_output_cost,
                'total_cost': total_cost,
            },
            'averages': {
                'avg_text_tokens': avg_text_tokens,
                'avg_image_tokens': avg_image_tokens,
                'avg_total_input_tokens': avg_total_input_tokens,
                'avg_output_tokens': avg_output_tokens,
                'avg_total_tokens': avg_total_tokens,
                'avg_images_per_conversation': avg_images_per_conversation,
                'avg_cost_per_conversation': avg_cost_per_conversation,
            },
            'per_conversation': results
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nDetailed results saved to: {args.output_file}")
        print(f"{'='*80}")
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

