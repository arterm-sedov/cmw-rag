#!/usr/bin/env python3
"""
Regex-only baseline evaluation on the 100-sample synthetic dataset.
"""

import json
import re
import time

DATASET_PATH = r"D:\Repo\cmw-rag\.opencode\plans\anonymization_test_dataset.json"

class CombinedRegexPatterns:
    RU_PHONE = re.compile(r'(\+7|8)[\s\-\(\)]?\d{3}[\s\-\(\)]?\d{3}[-\s]*\d{2}[-\s]*\d{2}')
    US_PHONE = re.compile(r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
    EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    IP = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    URL = re.compile(r'https?://[^\s<>\"]+')
    DATE = re.compile(r'\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2}')

    @classmethod
    def detect_all(cls, text: str) -> list[dict]:
        detections = []
        patterns = [
            (cls.EMAIL, 'EMAIL'), (cls.IP, 'IP_ADDRESS'), (cls.URL, 'URL'),
            (cls.RU_PHONE, 'PHONE'), (cls.US_PHONE, 'PHONE'), (cls.DATE, 'DATE')
        ]
        for pattern, etype in patterns:
            for m in pattern.finditer(text):
                detections.append({'text': m.group(), 'type': etype, 'start': m.start(), 'end': m.end()})
        return detections

# =============================================================================
# EVALUATION
# =============================================================================

# Map diverse types to common evaluation categories
EVAL_TYPE_MAP = {
    'LOCATION': 'ADDRESS',
    'CITY': 'ADDRESS',
    'STREET': 'ADDRESS',
    'COUNTRY': 'ADDRESS',
    'NAME': 'PERSON',
    'PERSON': 'PERSON',
    'COMPANY': 'ORGANIZATION',
    'ORGANIZATION': 'ORGANIZATION',
    'EMAIL': 'EMAIL',
    'PHONE': 'PHONE',
    'IP_ADDRESS': 'IP_ADDRESS',
    'URL': 'URL',
    'DATE': 'DATE'
}

def calculate_metrics(predicted, ground_truth):
    """Robust evaluation with normalized whitespace, case and types."""
    
    # Normalize ground truth: strip text, lower case, map types
    gt_set = set()
    for gt in ground_truth:
        norm_text = gt['text'].lower().strip()
        norm_type = EVAL_TYPE_MAP.get(gt['type'], gt['type'])
        gt_set.add((norm_text, norm_type))
    
    # Normalize predicted: strip text, lower case, map types
    pred_set = set()
    for p in predicted:
        norm_text = p['text'].lower().strip()
        norm_type = EVAL_TYPE_MAP.get(p['type'], p['type'])
        pred_set.add((norm_text, norm_type))
    
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    return tp, fp, fn

def main():
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    samples = dataset['samples']
    total_tp, total_fp, total_fn = 0, 0, 0
    start_time = time.time()
    for sample in samples:
        predicted = CombinedRegexPatterns.detect_all(sample['text'])
        tp, fp, fn = calculate_metrics(predicted, sample['entities'])
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    elapsed = time.time() - start_time
    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
