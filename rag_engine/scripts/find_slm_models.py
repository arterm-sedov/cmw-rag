"""
Find suitable open-source SLM models for PII extraction
Supports both EN and RU, available on OpenRouter and HuggingFace
"""

import requests
import json

def get_openrouter_models():
    """Fetch models from OpenRouter API."""
    response = requests.get("https://openrouter.ai/api/v1/models")
    data = response.json()
    
    models = []
    for model in data.get('data', []):
        model_id = model.get('id', '')
        name = model.get('name', '')
        context_length = model.get('context_length', 0)
        pricing = model.get('pricing', {})
        prompt_price = float(pricing.get('prompt', '0'))
        completion_price = float(pricing.get('completion', '0'))
        
        # Filter for open-source models
        if any(x in model_id.lower() for x in ['qwen', 'gpt-oss', 'gemma', 'mistral', 'llama', 'phi', 'deepseek']):
            models.append({
                'id': model_id,
                'name': name,
                'context': context_length,
                'prompt_price': prompt_price,
                'completion_price': completion_price
            })
    
    return models


def find_suitable_models():
    """Find models suitable for PII extraction (EN + RU)."""
    all_models = get_openrouter_models()
    
    # Group by size
    small_models = []  # <4B
    medium_models = []  # 4B-10B
    large_models = []  # 10B+
    
    for m in all_models:
        model_id = m['id']
        
        # Skip very large or very expensive
        if m['prompt_price'] > 0.001:  # Skip expensive models
            continue
        
        # Categorize by size
        if '0.5b' in model_id or '1.5b' in model_id or '1.7b' in model_id or '1.8b' in model_id:
            small_models.append(m)
        elif '2.7b' in model_id or '3b' in model_id or '4b' in model_id or '7b' in model_id or '8b' in model_id:
            medium_models.append(m)
        elif '14b' in model_id or '20b' in model_id or '27b' in model_id:
            large_models.append(m)
    
    print("=" * 80)
    print("SMALL MODELS (<4B) - Fast, Low VRAM")
    print("=" * 80)
    for m in sorted(small_models, key=lambda x: x['prompt_price']):
        print(f"{m['id']:<40} | ${m['prompt_price']:.6f}/1K | {m['context']} ctx")
    
    print("\n" + "=" * 80)
    print("MEDIUM MODELS (4B-10B) - Balanced")
    print("=" * 80)
    for m in sorted(medium_models, key=lambda x: x['prompt_price']):
        print(f"{m['id']:<40} | ${m['prompt_price']:.6f}/1K | {m['context']} ctx")
    
    print("\n" + "=" * 80)
    print("LARGE MODELS (10B+) - High Quality")
    print("=" * 80)
    for m in sorted(large_models, key=lambda x: x['prompt_price']):
        print(f"{m['id']:<40} | ${m['prompt_price']:.6f}/1K | {m['context']} ctx")
    
    # Recommend specific models for PII extraction
    print("\n" + "=" * 80)
    print("RECOMMENDED MODELS FOR PII EXTRACTION (EN + RU)")
    print("=" * 80)
    
    recommended = [
        ("qwen/qwen3-8b", "Best overall, 119 languages, tool calling, structured output"),
        ("qwen/qwen3-4b:free", "Free tier, good for testing, 119 languages"),
        ("qwen/qwen3-1.7b:free", "Smallest Qwen3, very fast, 119 languages"),
        ("openai/gpt-oss-20b", "Open-source GPT, good for comparison"),
        ("google/gemma-3-4b-it", "Gemma 4B, multilingual, good balance"),
        ("microsoft/phi-4", "Phi-4, efficient, good for structured output"),
    ]
    
    for model_id, desc in recommended:
        print(f"{model_id:<40} | {desc}")


if __name__ == "__main__":
    find_suitable_models()
