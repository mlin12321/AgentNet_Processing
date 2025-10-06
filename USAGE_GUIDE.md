# Quick Start Guide - AgentNet Token Calculator

## TL;DR - Get Started in 30 Seconds

```bash
# Activate environment
conda activate cua_data

# Run token calculator on full dataset
python token_calculator_pyspark.py --model gpt-4o

# View results
cat token_counts_detailed.json
```

## Common Use Cases

### 1. Calculate Costs for Different Models

```bash
# GPT-4o (balanced performance/cost)
python token_calculator_pyspark.py --model gpt-4o --output-file gpt4o_results.json

# GPT-4o-mini (cheapest, good performance)
python token_calculator_pyspark.py --model gpt-4o-mini --output-file gpt4o_mini_results.json

# o1-mini (reasoning model)
python token_calculator_pyspark.py --model o1-mini --output-file o1_mini_results.json

# o1-preview (advanced reasoning)
python token_calculator_pyspark.py --model o1-preview --output-file o1_preview_results.json
```

### 2. Compare Model Costs

```bash
# Compare two models
python compare_models_pyspark.py \
    -r gpt-4o gpt4o_results.json \
    -r gpt-4o-mini gpt4o_mini_results.json

# Compare all available results
python compare_models_pyspark.py --all
```

### 3. Debug/Development

```bash
# Process only 10 conversations with sample output
python token_calculator_pyspark.py --limit 10 --show-sample

# Test with specific model
python token_calculator_pyspark.py --model gpt-3.5-turbo --limit 5 --show-sample
```

## Understanding the Output

### Console Output

```
TOKEN TOTALS:
  Text tokens: 3,639,912        # From message text content
  Image tokens: 8,605,740       # From actual image files (based on dimensions)
  Total input tokens: 12,245,652
  Total output tokens: 21,006,832  # Fixed at 8,092 per conversation
  Total tokens: 33,252,484

PRICING:
  Input cost: $30.61            # Input tokens × model rate
  Output cost: $210.07          # Output tokens × model rate
  Total cost: $240.68           # Total for entire dataset
  Avg cost per conversation: $0.0927
```

### JSON Output Structure

```json
{
  "model": "gpt-4o",
  "totals": {
    "total_cost": 240.68,
    "total_input_tokens": 12245652
  },
  "averages": {
    "avg_cost_per_conversation": 0.0927
  },
  "per_conversation": [
    {
      "conversation_id": 0,
      "text_tokens": 993,
      "image_tokens": 3315,
      "total_tokens": 12400,
      "pricing": {
        "total_cost": 0.0917
      }
    }
  ]
}
```

## Key Differences from Older Versions

### ✅ What Changed

| Feature | Old Version | New PySpark Version |
|---------|-------------|---------------------|
| Image tokens | ~935 (approximated 5 tiles) | 1105 (actual calculation from image dimensions) |
| Text tokens | Approximation fallback | Always uses tiktoken |
| Pricing | Manual calculation | Official OpenAI rates built-in |
| Processing | Sequential | Parallel with PySpark |
| New models | None | o1, o1-mini, o1-preview, gpt-4o-2024-11-20 |

### Example: Image Token Calculation

**Old (Approximate)**:
```
Image tokens = 85 + (5 × 170) = 935 tokens
```

**New (Accurate)**:
```
1. Load image: 1920x1080
2. Scale to fit 2048x2048: 1920x1080 (no change)
3. Scale shortest to 768: 1365x768
4. Calculate tiles: ceil(1365/512) × ceil(768/512) = 3 × 2 = 6 tiles
5. Image tokens = 85 + (6 × 170) = 1105 tokens
```

## Model Selection Guide

### Cost vs. Performance

| Model | Cost/Conv* | Best For | Vision |
|-------|-----------|----------|--------|
| **gpt-4o-mini** | $0.006 | Budget-friendly, high volume | ✅ |
| **gpt-4o** | $0.093 | Balanced performance/cost | ✅ |
| **o1-mini** | $0.111 | Reasoning tasks, medium cost | ✅ |
| **gpt-4-turbo** | $0.293 | Legacy applications | ✅ |
| **o1** / **o1-preview** | $0.352 | Complex reasoning | ✅ |

*Based on avg 4,717 input + 8,092 output tokens per conversation

### Cost Comparison for Full Dataset (2,596 Conversations)

| Model | Total Cost | vs. Cheapest |
|-------|-----------|--------------|
| gpt-4o-mini | $14.54 | 1.00x |
| gpt-4o | $240.68 | 16.55x |
| o1-mini | $288.41 | 19.83x |
| o1-preview | $913.85 | 62.84x |
| gpt-4-turbo | $761.41 | 52.36x |

## Workflow Examples

### Example 1: Budget Planning

```bash
# Step 1: Calculate for cheapest model
python token_calculator_pyspark.py --model gpt-4o-mini --output-file budget.json

# Step 2: Calculate for target model
python token_calculator_pyspark.py --model gpt-4o --output-file target.json

# Step 3: Compare
python compare_models_pyspark.py -r mini budget.json -r standard target.json

# Result: See exact cost difference
```

### Example 2: Validate Dataset

```bash
# Quick sample to verify data format
python token_calculator_pyspark.py --limit 5 --show-sample

# Check for missing images or errors
grep "Warning" output.log
```

### Example 3: Production Estimation

```bash
# Calculate for all likely models
for model in gpt-4o gpt-4o-mini o1-mini; do
    python token_calculator_pyspark.py \
        --model $model \
        --output-file ${model}_production.json
done

# Compare all
python compare_models_pyspark.py --all
```

## Troubleshooting

### Issue: "Java gateway process exited"

**Solution**: Use the `cua_data` conda environment
```bash
conda activate cua_data
```

### Issue: "Image not found" warnings

**Cause**: Image paths in JSON don't match actual file locations

**Solution**: Check image paths or use `--limit` to skip problematic conversations

### Issue: Out of memory

**Solution**: Increase Spark memory in the script
```python
# In token_calculator_pyspark.py, line ~242
.set("spark.driver.memory", "64g")  # Increase from 32g
```

### Issue: Slow processing

**Normal Speed**: ~2,600 conversations in 15-20 seconds

**If slower**: 
- Check system resources
- Reduce parallelism with `.setMaster("local[4]")` instead of `local[*]`

## FAQ

**Q: Which model should I use for production?**

A: For most cases, `gpt-4o-mini` offers the best cost/performance. Use `gpt-4o` if you need higher quality, or `o1-mini` for reasoning tasks.

**Q: Are the token counts exact?**

A: Yes. Text tokens use tiktoken (OpenAI's official library). Image tokens use OpenAI's documented tiling algorithm with actual image dimensions.

**Q: Can I add my own model?**

A: Yes! Edit `OPENAI_MODELS` dict in `token_calculator_pyspark.py`. See README for details.

**Q: Why are image tokens 1105 instead of ~935?**

A: The new version calculates actual tiles based on image dimensions. Most images in the dataset are 1920x1080, which equals 6 tiles = 1105 tokens.

**Q: What's the difference between the files?**

- `token_calculator_pyspark.py` - ✅ **Use this** (accurate, fast, PySpark)
- `token_calculator_simple.py` - Legacy (no PySpark, slower)
- `token_calculator.py` - Deprecated draft

## Advanced: Extending the Calculator

### Add Custom Pricing Tier

```python
# In token_calculator_pyspark.py
OPENAI_MODELS['custom-model'] = {
    'encoding': 'o200k_base',
    'supports_vision': True,
    'input_price_per_1m': 1.00,
    'output_price_per_1m': 5.00,
}
```

### Batch Process Multiple Datasets

```bash
for dataset in dataset1.json dataset2.json dataset3.json; do
    python token_calculator_pyspark.py \
        --input-file $dataset \
        --output-file ${dataset%.json}_tokens.json
done
```

### Export to CSV

```python
import json
import pandas as pd

with open('token_counts_detailed.json') as f:
    data = json.load(f)

df = pd.DataFrame(data['per_conversation'])
df.to_csv('token_analysis.csv', index=False)
```

## Performance Benchmarks

Hardware: Standard workstation with 32GB RAM

| Dataset Size | Processing Time | Throughput |
|-------------|-----------------|------------|
| 10 conversations | 2s | 5 conv/s |
| 100 conversations | 4s | 25 conv/s |
| 2,596 conversations | 15s | 173 conv/s |

## Support

For issues or questions:
1. Check this guide
2. Review README.md
3. Check script comments
4. Verify conda environment is activated
