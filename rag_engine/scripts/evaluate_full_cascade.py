#!/usr/bin/env python3
"""
Full pipeline evaluation on the 100-sample synthetic dataset.
Cascade: Regex + dslim + Gherman + EU-PII
"""

import json
import time
import re
from typing import Any
from transformers import pipeline

# Configuration
DATASET_PATH = r"D:\Repo\cmw-rag\.opencode\plans\anonymization_test_dataset.json"

# =============================================================================
# REGEX DETECTION (US + RU)
# =============================================================================

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
                detections.append({'text': m.group(), 'type': etype, 'start': m.start(), 'end': m.end(), 'source': 'regex'})
        return detections

# =============================================================================
# NER MODELS
# =============================================================================

_model_cache = {}

def get_ner_pipeline(model_name):
    if model_name not in _model_cache:
        print(f"Loading {model_name}...")
        _model_cache[model_name] = pipeline("ner", model=model_name, aggregation_strategy="simple")
    return _model_cache[model_name]

MODELS = {
    'dslim': 'dslim/bert-large-NER',
    'gherman': 'Gherman/bert-base-NER-Russian',
    'eu_pii': 'tabularisai/eu-pii-safeguard'
}

ENTITY_MAPS = {
    'dslim': {'PER': 'PERSON', 'ORG': 'ORGANIZATION', 'LOC': 'LOCATION'},
    'gherman': {'PER': 'PERSON', 'FIRST_NAME': 'PERSON', 'LAST_NAME': 'PERSON', 'ORG': 'ORGANIZATION', 'CITY': 'LOCATION', 'STREET': 'ADDRESS'},
    'eu_pii': {'PERSON': 'PERSON', 'ORGANIZATION': 'ORGANIZATION', 'EMAIL_ADDRESS': 'EMAIL', 'PHONE_NUMBER': 'PHONE'}
}

def run_ner(text, model_key):
    try:
        pipe = get_ner_pipeline(MODELS[model_key])
        results = pipe(text)
        detections = []
        label_map = ENTITY_MAPS[model_key]
        for r in results:
            label = r.get('entity_group', '')
            if label in label_map and r.get('score', 0) > 0.4:
                detections.append({
                    'text': r.get('word', ''),
                    'type': label_map[label],
                    'start': r.get('start', 0),
                    'end': r.get('end', 0),
                    'confidence': float(r.get('score', 0)),
                    'source': model_key
                })
        return detections
    except Exception as e:
        print(f"Error running {model_key}: {e}")
        return []

# =============================================================================
# MERGING & FRAGMENTATION
# =============================================================================

# Domain-specific block-list for IT support
TECH_BLOCKLIST = {
    "elasticsearch", "comindware", "postgresql", "kafka", "mysql", "oracle", "linux", 
    "architect", "apigateway", "service", "admin", "support", "systemaccount", 
    "root", "localhost", "nginx", "docker", "kubernetes", "platform", "ubuntu",
    "windows", "office", "excel", "outlook", "smtp", "imap", "vpn", "firewall",
    "substep", "logics", "think", "impl", "transaction", "commit", "context"
}

def is_technical_noise(text: str, etype: str) -> bool:
    """Check if an entity is likely technical noise rather than PII."""
    clean = text.lower().strip()
    
    # 1. Block-list check
    if clean in TECH_BLOCKLIST:
        return True
        
    # 2. Structural checks for PERSON/ORGANIZATION
    if etype in ["PERSON", "ORGANIZATION"]:
        # Service names often have dots, underscores or numbers
        if any(c in text for c in "._/\\0123456789"):
            return True
        # CamelCase noise (e.g. systemAccount, architectService)
        if re.search(r'[a-z][A-Z]', text) and not re.search(r'\s', text):
            return True
        # All-caps noise (e.g. ERROR, WARN, INFO)
        if text.isupper() and len(text) > 3 and not re.search(r'\s', text):
            return True
            
    return False

def merge_and_dedupe(entities, text):
    if not entities: return []
    
    # Pre-filter noise
    filtered_entities = []
    for e in entities:
        # Filter BERT fragments
        clean = e['text'].replace('##', '').strip()
        if len(clean) <= 2 and e['source'] != 'regex': 
            continue
            
        # Filter domain-specific noise (NEW)
        if is_technical_noise(e['text'], e['type']):
            continue
            
        filtered_entities.append(e)

    # Sort by start, then length (longer first)
    filtered_entities.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))
    
    final = []
    covered = [False] * len(text)
    for e in filtered_entities:
        # Check overlaps
        if not any(covered[i] for i in range(e['start'], e['end'])):
            final.append(e)
            for i in range(e['start'], e['end']): 
                covered[i] = True
                
    return sorted(final, key=lambda x: x['start'])

# =============================================================================
# EVALUATION
# =============================================================================

# Map diverse NER types to common evaluation categories
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

    print(f"Evaluating 4-stage cascade on {len(samples)} samples...")
    
    for i, sample in enumerate(samples):
        text = sample['text']
        gt = sample['entities']
        
        # Cascade
        all_detections = []
        all_detections.extend(CombinedRegexPatterns.detect_all(text))
        
        # Smart routing
        cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
        if cyrillic > 5:
            all_detections.extend(run_ner(text, 'gherman'))
        else:
            all_detections.extend(run_ner(text, 'dslim'))
            
        all_detections.extend(run_ner(text, 'eu_pii'))
        
        predicted = merge_and_dedupe(all_detections, text)
        tp, fp, fn = calculate_metrics(predicted, gt)
        
        # DEBUG: Print False Positives for the first 10 samples
        if i < 10:
            gt_set = set((gt_e['text'].lower(), gt_e['type']) for gt_e in gt)
            pred_set = set((p['text'].lower(), p['type']) for p in predicted)
            fps = pred_set - gt_set
            fns = gt_set - pred_set
            if fps or fns:
                print(f"Sample {i+1} FPS: {fps}")
                print(f"Sample {i+1} FNS: {fns}")
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(samples)}...")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Total Samples: {len(samples)}")
    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Total Time: {elapsed:.1f}s ({elapsed/len(samples)*1000:.1f}ms/sample)")
    print("="*50)

if __name__ == "__main__":
    main()
