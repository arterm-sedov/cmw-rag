#!/usr/bin/env python3
"""
Comprehensive US/English NER model comparison test.
Tests multiple cascade configurations and NER models.
"""

import time
import re
from dataclasses import dataclass
from typing import Any

# =============================================================================
# TEST DATA - English/US business/IT/PII samples
# =============================================================================

ENGLISH_SAMPLES = [
    # Business emails
    "Please contact John Smith at john.smith@company.com or call +1-555-123-4567 for the project deliverables.",
    "Our IT department needs to reset the password for user admin@corp.local. IP: 192.168.1.100",
    "The server credentials are: username: administrator, password: SuperSecret123! at 10.0.0.55",
    
    # Personal info in business context
    "Employee Maria Garcia (SSN: 123-45-6789) from Los Angeles office, phone: +1-555-987-6543 needs access.",
    "John Doe, DOB: 03/15/1985, email: john.doe@gmail.com, credit card: 4532-1234-5678-9012 for subscription.",
    
    # IT infrastructure
    "VPN access required for 172.16.0.50. API endpoint: https://api.internal.corp.local/v1/users. Key: sk_live_abc123xyz",
    "Database credentials: user=db_admin, pass=P@ssw0rd! located at 192.168.10.25:5432",
    
    # Customer data
    "Customer Robert Johnson - Email: r.johnson@customer.net, Phone: +1-555-111-2222, Address: 123 Main St, Boston MA 02101",
    "Invoice to: Sarah Williams, CC: sarah.w@business.org, ABN: 12345678901, Phone: +61-2-1234-5678",
    
    # Medical/Insurance (English)
    "Patient Michael Brown, DOB: 05/22/1978, Medicare: 123456789012, Insurance ID: MED-9876543, email: mbrown@email.com",
    
    # Additional US-specific
    "Dr. Sarah Johnson, NPI: 1234567890, DEA: AB1234567, License: CA123456, EIN: 12-3456789",
    "Shipping address: 456 Oak Avenue, Suite 200, Chicago IL 60601. ZIP: 60601-1234",
]

# =============================================================================
# US REGEX PATTERNS
# =============================================================================

class USRegexPatterns:
    """US-specific regex patterns for English PII."""
    
    PHONE_PATTERNS = [
        re.compile(r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
    ]
    
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    ITIN_PATTERN = re.compile(r'\b9\d{2}-\d{7}\b')
    EIN_PATTERN = re.compile(r'\b\d{2}-\d{7}\b')
    ZIP_PATTERN = re.compile(r'\b\d{5}(-\d{4})?\b')
    DRIVER_LICENSE_PATTERN = re.compile(r'\b[A-Z]{1,2}\d{5,8}\b')
    DEA_PATTERN = re.compile(r'\b[A-Z]{2}\d{7}\b')
    NPI_PATTERN = re.compile(r'\b[1-2]\d{9}\b')
    
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
    IP_PATTERN = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    DOB_PATTERN = re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b')
    
    PASSWORD_PATTERN = re.compile(r'(password|passwd|pwd|pass)[\s:=_]+(\S+)', re.IGNORECASE)
    USERNAME_PATTERN = re.compile(r'(username|user|login)[\s:=_]+(\S+)', re.IGNORECASE)
    API_KEY_PATTERN = re.compile(r'(api[_-]?key|apikey)[\s:=_]+(\S+)', re.IGNORECASE)
    INTERNAL_URL_PATTERN = re.compile(r'(https?)://([\w.-]+internal|[\w.-]+local|192\.168|10\.|172\.(1[6-9]|2|3[01]))[^\s]*', re.IGNORECASE)
    
    @classmethod
    def detect_all(cls, text: str) -> list[dict]:
        """Detect all PII using US regex patterns."""
        detections = []
        
        def add_detection(pattern, ptype, confidence=0.95):
            for match in pattern.finditer(text):
                overlaps = any(
                    not (match.end() <= d['start'] or match.start() >= d['end'])
                    for d in detections
                )
                if not overlaps:
                    detections.append({
                        'type': ptype,
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': confidence,
                        'source': 'us_regex'
                    })
        
        add_detection(cls.EMAIL_PATTERN, 'EMAIL')
        add_detection(cls.SSN_PATTERN, 'US_SSN')
        add_detection(cls.ITIN_PATTERN, 'US_ITIN')
        add_detection(cls.EIN_PATTERN, 'US_EIN')
        add_detection(cls.NPI_PATTERN, 'US_NPI')
        add_detection(cls.CREDIT_CARD_PATTERN, 'CREDIT_CARD')
        add_detection(cls.IP_PATTERN, 'IP')
        
        for p in cls.PHONE_PATTERNS:
            add_detection(p, 'PHONE')
        
        add_detection(cls.PASSWORD_PATTERN, 'PASSWORD')
        add_detection(cls.USERNAME_PATTERN, 'USERNAME')
        add_detection(cls.API_KEY_PATTERN, 'API_KEY')
        add_detection(cls.INTERNAL_URL_PATTERN, 'INTERNAL_URL')
        add_detection(cls.DOB_PATTERN, 'DATE')
        add_detection(cls.ZIP_PATTERN, 'US_ZIP')
        add_detection(cls.DRIVER_LICENSE_PATTERN, 'US_DRIVER_LICENSE')
        add_detection(cls.DEA_PATTERN, 'US_DEA')
        
        return sorted(detections, key=lambda x: x['start'])


# =============================================================================
# NER MODEL LOADERS
# =============================================================================

def load_model_cached(name, **kwargs):
    """Load model with caching."""
    if not hasattr(load_model_cached, '_cache'):
        load_model_cached._cache = {}
    
    if name in load_model_cached._cache:
        return load_model_cached._cache[name]
    
    try:
        from transformers import pipeline
        model = pipeline("ner", model=name, tokenizer=name, aggregation_strategy="simple", **kwargs)
        load_model_cached._cache[name] = model
        print(f"  Loaded: {name}")
        return model
    except Exception as e:
        print(f"  Failed to load {name}: {e}")
        return None


MODELS = {
    'gherman': 'Gherman/bert-base-NER-Russian',
    'eu_pii': 'tabularisai/eu-pii-safeguard',
    'dslim_bert': 'dslim/bert-base-NER',
    'dslim_bert_large': 'dslim/bert-large-NER',
    'roberta_large': 'Jean-Baptiste/roberta-large-ner-english',
    'flair': 'flair/ner-english',
    'protectai': 'protectai/bert-base-NER-onnx',
}


# =============================================================================
# ENTITY MAPPINGS FOR EACH MODEL
# =============================================================================

ENTITY_MAPS = {
    'gherman': {
        'PER': 'PERSON', 'PERSON': 'PERSON',
        'LOC': 'LOCATION', 'LOCATION': 'LOCATION',
        'ORG': 'ORGANIZATION', 'ORGANIZATION': 'ORGANIZATION',
    },
    'eu_pii': {
        'PERSON': 'PERSON', 'EMAIL_ADDRESS': 'EMAIL',
        'PHONE_NUMBER': 'PHONE', 'LOCATION': 'LOCATION',
        'ORGANIZATION': 'ORGANIZATION',
    },
    'dslim_bert': {
        'PER': 'PERSON', 'LOC': 'LOCATION', 'ORG': 'ORGANIZATION', 'MISC': 'MISC',
    },
    'dslim_bert_large': {
        'PER': 'PERSON', 'LOC': 'LOCATION', 'ORG': 'ORGANIZATION', 'MISC': 'MISC',
    },
    'roberta_large': {
        'PER': 'PERSON', 'LOC': 'LOCATION', 'ORG': 'ORGANIZATION', 'MISC': 'MISC',
    },
    'flair': {
        'PER': 'PERSON', 'LOC': 'LOCATION', 'ORG': 'ORGANIZATION', 'MISC': 'MISC',
    },
    'protectai': {
        'PER': 'PERSON', 'LOC': 'LOCATION', 'ORG': 'ORGANIZATION', 'MISC': 'MISC',
    },
}


def run_model(text: str, model_key: str) -> list[dict]:
    """Run a specific NER model on text."""
    model = load_model_cached(MODELS[model_key], device='cpu')
    if not model:
        return []
    
    results = model(text)
    detections = []
    label_map = ENTITY_MAPS.get(model_key, {})
    
    for r in results:
        label = r.get('entity_group', '')
        if label in label_map and r.get('score', 0) > 0.5:
            detections.append({
                'type': label_map[label],
                'text': r.get('word', ''),
                'start': r.get('start', 0),
                'end': r.get('end', 0),
                'confidence': r.get('score', 0),
                'source': model_key
            })
    
    return detections


# =============================================================================
# MERGE FUNCTION
# =============================================================================

def merge_and_dedupe(all_entities: list[dict], text: str) -> list[dict]:
    """Merge overlapping entities, keep longest."""
    if not all_entities:
        return []
    
    sorted_ents = sorted(all_entities, key=lambda x: (x['end'] - x['start'], -x['start']), reverse=True)
    
    covered = [False] * len(text)
    final = []
    
    for e in sorted_ents:
        start, end = e['start'], e['end']
        overlaps = any(covered[i] for i in range(start, min(end, len(text))))
        
        if not overlaps:
            final.append(e)
            for i in range(start, min(end, len(text))):
                covered[i] = True
    
    return sorted(final, key=lambda x: x['start'])


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_single_model(model_key: str, samples: list[str]) -> dict:
    """Test a single NER model."""
    total_time = 0
    total_detections = 0
    
    for sample in samples:
        t0 = time.time()
        results = run_model(sample, model_key)
        total_time += time.time() - t0
        total_detections += len(results)
    
    return {
        'model': model_key,
        'total_time': total_time,
        'avg_time_ms': (total_time / len(samples)) * 1000,
        'total_detections': total_detections,
        'avg_detections': total_detections / len(samples),
    }


def test_cascade(cascade_name: str, stages: list[str], samples: list[str]) -> dict:
    """Test a cascade of stages."""
    total_time = 0
    total_detections = 0
    
    for sample in samples:
        all_entities = []
        
        # Stage 1: US Regex
        if 'us_regex' in stages:
            t0 = time.time()
            all_entities.extend(USRegexPatterns.detect_all(sample))
            total_time += time.time() - t0
        
        # NER stages
        for stage in stages:
            if stage not in ['us_regex', 'eu_pii', 'gherman', 'dslim', 'dslim_large', 'roberta', 'flair']:
                continue
            
            model_key = stage.replace('dslim', 'dslim_bert').replace('dslim_large', 'dslim_bert_large').replace('roberta', 'roberta_large').replace('flair', 'flair')
            t0 = time.time()
            all_entities.extend(run_model(sample, model_key))
            total_time += time.time() - t0
        
        merged = merge_and_dedupe(all_entities, sample)
        total_detections += len(merged)
    
    return {
        'cascade': cascade_name,
        'total_time': total_time,
        'avg_time_ms': (total_time / len(samples)) * 1000,
        'total_detections': total_detections,
        'avg_detections': total_detections / len(samples),
    }


def main():
    print("=" * 80)
    print("US/ENGLISH NER MODEL COMPARISON")
    print("=" * 80)
    print(f"\nTesting {len(ENGLISH_SAMPLES)} samples\n")
    
    # Test each NER model individually
    print("\n" + "=" * 60)
    print("SINGLE MODEL BENCHMARKS")
    print("=" * 60)
    
    results = []
    for model_key in ['dslim_bert', 'dslim_bert_large', 'roberta_large', 'flair', 'protectai', 'eu_pii', 'gherman']:
        print(f"\nTesting {model_key}...")
        r = test_single_model(model_key, ENGLISH_SAMPLES)
        results.append(r)
        print(f"  Time: {r['avg_time_ms']:.1f}ms/sample, Detections: {r['avg_detections']:.1f}/sample")
    
    # Sort by time
    results.sort(key=lambda x: x['avg_time_ms'])
    
    print("\n" + "=" * 60)
    print("RANKED BY SPEED")
    print("=" * 60)
    for r in results:
        print(f"  {r['model']:20s}: {r['avg_time_ms']:6.1f}ms/sample, {r['avg_detections']:.1f} detections")
    
    # Test cascades
    print("\n" + "=" * 60)
    print("CASCADE BENCHMARKS")
    print("=" * 60)
    
    cascades = [
        ("US Regex only", ["us_regex"]),
        ("EU-PII only", ["eu_pii"]),
        ("dslim only", ["dslim"]),
        ("US Regex + dslim", ["us_regex", "dslim"]),
        ("US Regex + EU-PII", ["us_regex", "eu_pii"]),
        ("US Regex + dslim + EU-PII", ["us_regex", "dslim", "eu_pii"]),
        ("US Regex + dslim + EU-PII + Gherman", ["us_regex", "dslim", "eu_pii", "gherman"]),
    ]
    
    cascade_results = []
    for name, stages in cascades:
        print(f"\nTesting cascade: {name}")
        r = test_cascade(name, stages, ENGLISH_SAMPLES)
        cascade_results.append(r)
        print(f"  Time: {r['avg_time_ms']:.1f}ms/sample, Detections: {r['avg_detections']:.1f}/sample")
    
    # Show detailed results for best cascades
    print("\n" + "=" * 60)
    print("BEST CASCADES (by detections)")
    print("=" * 60)
    cascade_results.sort(key=lambda x: -x['avg_detections'])
    for r in cascade_results[:5]:
        print(f"  {r['cascade']:40s}: {r['avg_time_ms']:6.1f}ms, {r['avg_detections']:.1f} dets")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
