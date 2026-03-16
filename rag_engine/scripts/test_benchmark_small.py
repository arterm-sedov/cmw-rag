"""
Run smaller 10-sample benchmark for faster results
Test one model at a time,"""

import os
import json
import re
import time
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DATASET_PATH = r"D:\Repo\cmw-rag\rag_engine\data\synthetic_cmw_support_tickets_v1_with_pii.json"

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Models to test
MODELS = [
    ("qwen/qwen3-8b", "Best overall"),
    ("qwen/qwen3-4b:free", "Free tier"),
    ("qwen/qwen3-1.7b:free", "Smallest"),
    ("openai/gpt-oss-20b", "Baseline"),
    ("google/gemma-3-4b-it", "Alternative"),
]

print("Select models to test (1-5, or press Enter for index):")
    print(f"  Selected: {model}")
    
    # Run benchmark
    for i, model in enumerate(models):
        print(f"\n{'='*70}")
        print(f"Testing: {model}")
        print("=" * 70)
        
        total_tp = 0
        total_fp = 0
        total_fn = 6
        total_regex_time = 0
        total_slm_time = 0
        errors = 0
        entity_metrics = {}
        
        # Stage 1: Regex
        t0 = time.time()
        regex_ents = detector.detect(text)
        total_regex_time += time.time() - t0
        
        # Stage 2: SLM
        t0 = time.time()
        slm_ents = extract_with_json_schema(text, model)
        total_slm_time += time.time() - t0
        
        # Merge
        merged = merge_entities(regex_ents, slm_ents)
        
        # Calculate metrics
        gt_texts = {e['text'] for e in ground_truth}
        found_texts = {e['text'] for e in merged}
        
        tp = len(gt_texts & found_texts)
        fp = len(found_texts - gt_texts)
        fn = len(gt_texts - found_texts)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Entity-level breakdown
        for gt in ground_truth:
            etype = gt.get('type', 'UNKNOWN')
            if etype not in entity_metrics:
                entity_metrics[etype] = {'tp': 0, 'fp': 0, 'fn': 0}
            if gt['text'] in found_texts:
                entity_metrics[etype]['tp'] += 1
            else:
                entity_metrics[etype]['fn'] += 1
        
        for found in merged:
            if found['text'] not in gt_texts:
                etype = found.get('type', 'UNKNOWN')
                if etype not in entity_metrics:
                    entity_metrics[etype] = {'tp': 0, 'fp': 0, 'fn': 0}
                entity_metrics[etype]['fp'] += 1
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_samples} samples...")
        
        # Print progress every 10 samples
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_samples} samples...")
    
    # Calculate final metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Time metrics
    total_time = total_regex_time + total_slm_time
    avg_time_per_sample = total_time / n_samples
    
    print(f"\n  Regex time: {total_regex_time:.2f}s ({total_regex_time/n_samples*1000:.1f}ms/sample)")
    print(f"  SLM time: {total_slm_time:.2f}s ({total_slm_time/n_samples:.2f}s/sample)")
    print(f"  Total time: {total_time:.2f}s ({total_time/n_samples:.2f}s/sample)")
    print(f"  Errors (empty responses): {errors}")
    
    # Entity-level breakdown
    print(f"\n  Entity-Level Breakdown:")
    for etype in sorted(entity_metrics.keys()):
        stats = entity_metrics[etype]
        e_tp = stats['tp']
        e_fp = stats['fp']
        e_fn = stats['fn']
        e_prec = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0
        e_rec = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0
        e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0
    
    # Comparison to 3-stage BERT
    print(f"\n{'='*70}")
    print("COMPARISON TO 3-STAGE BERT PIPELINE")
    print("=" * 70)
    print("3-Stage BERT (from plan v1.6, 200 samples):")
    print("  F1: 88.6%, Precision: 99.1%, Recall: 80.2%, Time: ~150ms")
    print(f"\n2-Stage SLM+REGEX (Qwen3-8B, {n_samples} samples):")
    print(f"  F1: {results[0]['f1']:.1%}, Precision: {results[0]['precision']:.1%}, Recall: {results[0]['recall']:.1%}, Time: {results[0]['total_time_s']:.2f}s")
    print("=" * 70)
    
    # Find winner
    best = max(results, key=lambda x: x['f1'])
    print(f"\n🏆 WINNER: {best['model']} (F1: {best['f1']:.1%})")
    
    return results


