#!/usr/bin/env python3
"""
Ad-hoc comparison: Presidio+Stage2+Stage3 vs Regex+Stage2+Stage3 cascade
for English business/IT/PII samples.
"""

import time
import re
from dataclasses import dataclass
from typing import Any

# =============================================================================
# TEST DATA - English business/IT/PII samples
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
]

# =============================================================================
# STAGE 1: Direct Regex Patterns (our approach)
# =============================================================================

class OurRegexPatterns:
    """Our direct regex patterns for English PII."""
    
    PHONE_PATTERNS = [
        re.compile(r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),  # US/Canada
        re.compile(r'\+?\d{1,3}[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}'),  # International
    ]
    
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    
    # Financial IDs
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
    ABN_PATTERN = re.compile(r'\b\d{11}\b')  # Australian Business Number
    
    # IP Addresses
    IP_PATTERN = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    
    # Dates
    DOB_PATTERN = re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b')
    
    # Corporate sensitive
    PASSWORD_PATTERN = re.compile(r'(password|passwd|pwd|pass)[\s:=_]+(\S+)', re.IGNORECASE)
    USERNAME_PATTERN = re.compile(r'(username|user|login)[\s:=_]+(\S+)', re.IGNORECASE)
    API_KEY_PATTERN = re.compile(r'(api[_-]?key|apikey)[\s:=_]+(\S+)', re.IGNORECASE)
    INTERNAL_URL_PATTERN = re.compile(r'(https?)://([\w.-]+internal|[\w.-]+local|192\.168|10\.|172\.(1[6-9]|2|3[01]))[^\s]*', re.IGNORECASE)
    
    @classmethod
    def detect_all(cls, text: str) -> list[dict]:
        """Detect all PII using regex patterns."""
        detections = []
        
        def add_detection(pattern, ptype, confidence=0.95):
            for match in pattern.finditer(text):
                # Check overlap
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
                        'source': 'regex'
                    })
        
        # High confidence patterns first
        add_detection(cls.EMAIL_PATTERN, 'EMAIL')
        add_detection(cls.SSN_PATTERN, 'SSN')
        add_detection(cls.CREDIT_CARD_PATTERN, 'CREDIT_CARD')
        add_detection(cls.IP_PATTERN, 'IP')
        
        for p in cls.PHONE_PATTERNS:
            add_detection(p, 'PHONE')
        
        add_detection(cls.PASSWORD_PATTERN, 'PASSWORD')
        add_detection(cls.USERNAME_PATTERN, 'USERNAME')
        add_detection(cls.API_KEY_PATTERN, 'API_KEY')
        add_detection(cls.INTERNAL_URL_PATTERN, 'INTERNAL_URL')
        add_detection(cls.DOB_PATTERN, 'DATE')
        add_detection(cls.ABN_PATTERN, 'ABN')
        
        return sorted(detections, key=lambda x: x['start'])


# =============================================================================
# PRESIDIO APPROACH (simulated - would need actual Presidio install)
# =============================================================================

class PresidioSimulator:
    """Simulates what Presidio would detect."""
    
    # Presidio's built-in recognizers for English
    PRESIDIO_PATTERNS = {
        'EMAIL_ADDRESS': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
        'PHONE_NUMBER': re.compile(r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
        'CREDIT_CARD': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        'IP_ADDRESS': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        'US_SSN': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        'DATE_TIME': re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b'),
    }
    
    @classmethod
    def detect_all(cls, text: str) -> list[dict]:
        """Simulate Presidio detection."""
        detections = []
        
        for ptype, pattern in cls.PRESIDIO_PATTERNS.items():
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
                        'confidence': 0.85,  # Presidio typically reports ~0.85
                        'source': 'presidio'
                    })
        
        return sorted(detections, key=lambda x: x['start'])


# =============================================================================
# STAGE 2 & 3: ML NER Models (Gherman + EU-PII)
# =============================================================================

def load_gherman():
    """Load Gherman model for English entity detection."""
    try:
        from transformers import pipeline
        return pipeline(
            "ner",
            model="Gherman/bert-base-NER-Russian",
            tokenizer="Gherman/bert-base-NER-Russian",
            aggregation_strategy="simple",
            device="cpu",
        )
    except Exception as e:
        print(f"  Gherman not available: {e}")
        return None


def load_eu_pii():
    """Load EU-PII-Safeguard model."""
    try:
        from transformers import pipeline
        return pipeline(
            "ner",
            model="tabularisai/eu-pii-safeguard",
            tokenizer="tabularisai/eu-pii-safeguard",
            aggregation_strategy="simple",
            device="cpu",
        )
    except Exception as e:
        print(f"  EU-PII not available: {e}")
        return None


def run_gherman(text: str, model) -> list[dict]:
    """Run Gherman on text."""
    if not model:
        return []
    
    results = model(text)
    detections = []
    
    # Gherman labels (works partially on English)
    label_map = {
        'PER': 'PERSON', 'PERSON': 'PERSON',
        'LOC': 'LOCATION', 'LOCATION': 'LOCATION',
        'ORG': 'ORGANIZATION', 'ORGANIZATION': 'ORGANIZATION',
    }
    
    for r in results:
        label = r.get('entity_group', '')
        if label in label_map and r.get('score', 0) > 0.5:
            detections.append({
                'type': label_map[label],
                'text': r.get('word', ''),
                'start': r.get('start', 0),
                'end': r.get('end', 0),
                'confidence': r.get('score', 0),
                'source': 'gherman'
            })
    
    return detections


def run_eu_pii(text: str, model) -> list[dict]:
    """Run EU-PII-Safeguard on text."""
    if not model:
        return []
    
    results = model(text)
    detections = []
    
    # EU-PII labels for English
    label_map = {
        'PERSON': 'PERSON',
        'EMAIL_ADDRESS': 'EMAIL',
        'PHONE_NUMBER': 'PHONE',
        'LOCATION': 'LOCATION',
        'ORGANIZATION': 'ORGANIZATION',
    }
    
    for r in results:
        label = r.get('entity_group', '')
        if label in label_map and r.get('score', 0) > 0.5:
            detections.append({
                'type': label_map[label],
                'text': r.get('word', ''),
                'start': r.get('start', 0),
                'end': r.get('end', 0),
                'confidence': r.get('score', 0),
                'source': 'eu_pii'
            })
    
    return detections


# =============================================================================
# CASCADE PIPELINES
# =============================================================================

def merge_and_dedupe(all_entities: list[dict], text: str) -> list[dict]:
    """Merge overlapping entities, keep longest."""
    if not all_entities:
        return []
    
    # Sort by length (longest first)
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
    
    # Sort by position
    return sorted(final, key=lambda x: x['start'])


def run_presidio_cascade(text: str, gherman_model, eu_pii_model) -> list[dict]:
    """Presidio + Gherman + EU-PII cascade."""
    all_entities = []
    
    # Stage 1: Presidio
    presidio_results = PresidioSimulator.detect_all(text)
    all_entities.extend(presidio_results)
    
    # Stage 2: Gherman
    gherman_results = run_gherman(text, gherman_model)
    all_entities.extend(gherman_results)
    
    # Stage 3: EU-PII
    eu_pii_results = run_eu_pii(text, eu_pii_model)
    all_entities.extend(eu_pii_results)
    
    return merge_and_dedupe(all_entities, text)


def run_our_cascade(text: str, gherman_model, eu_pii_model) -> list[dict]:
    """Regex + Gherman + EU-PII cascade (our approach)."""
    all_entities = []
    
    # Stage 1: Our Regex
    regex_results = OurRegexPatterns.detect_all(text)
    all_entities.extend(regex_results)
    
    # Stage 2: Gherman
    gherman_results = run_gherman(text, gherman_model)
    all_entities.extend(gherman_results)
    
    # Stage 3: EU-PII
    eu_pii_results = run_eu_pii(text, eu_pii_model)
    all_entities.extend(eu_pii_results)
    
    return merge_and_dedupe(all_entities, text)


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def main():
    print("=" * 80)
    print("COMPARISON: Presidio+Stage2+Stage3 vs Regex+Stage2+Stage3")
    print("=" * 80)
    print(f"\nTesting {len(ENGLISH_SAMPLES)} English samples\n")
    
    # Load models
    print("Loading ML models...")
    gherman = load_gherman()
    eu_pii = load_eu_pii()
    print("Models loaded.\n")
    
    total_presidio = 0
    total_our_regex = 0
    total_presidio_full = 0
    total_our_full = 0
    
    for i, sample in enumerate(ENGLISH_SAMPLES):
        print(f"{'='*60}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*60}")
        print(f"Text: {sample[:80]}...")
        print()
        
        # Presidio only
        t0 = time.time()
        presidio_only = PresidioSimulator.detect_all(sample)
        presidio_time = time.time() - t0
        total_presidio += presidio_time
        
        # Our Regex only
        t0 = time.time()
        regex_only = OurRegexPatterns.detect_all(sample)
        regex_time = time.time() - t0
        total_our_regex += regex_time
        
        # Full cascade: Presidio + Stage2 + Stage3
        t0 = time.time()
        presidio_cascade = run_presidio_cascade(sample, gherman, eu_pii)
        presidio_full_time = time.time() - t0
        total_presidio_full += presidio_full_time
        
        # Full cascade: Regex + Stage2 + Stage3 (our approach)
        t0 = time.time()
        our_cascade = run_our_cascade(sample, gherman, eu_pii)
        our_full_time = time.time() - t0
        total_our_full += our_full_time
        
        print(f"--- STAGE 1 COMPARISON ---")
        print(f"Presidio:  {len(presidio_only):2d} detections in {presidio_time*1000:6.1f}ms")
        print(f"Our Regex: {len(regex_only):2d} detections in {regex_time*1000:6.1f}ms")
        
        print(f"\n--- FULL CASCADE COMPARISON ---")
        print(f"Presidio+St2+St3: {len(presidio_cascade):2d} detections in {presidio_full_time*1000:6.1f}ms")
        print(f"Regex+St2+St3:    {len(our_cascade):2d} detections in {our_full_time*1000:6.1f}ms")
        
        # Show our cascade results
        print(f"\n--- OUR CASCADE DETAILS ---")
        for e in our_cascade:
            print(f"  [{e['source']:8s}] {e['type']:15s}: {e['text'][:40]}")
        
        print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nStage 1 Only:")
    print(f"  Presidio:  {total_presidio*1000:7.1f}ms total, {total_presidio/len(ENGLISH_SAMPLES)*1000:.2f}ms/sample")
    print(f"  Our Regex:  {total_our_regex*1000:7.1f}ms total, {total_our_regex/len(ENGLISH_SAMPLES)*1000:.2f}ms/sample")
    
    print(f"\nFull Cascade (Stage 1 + Stage 2 + Stage 3):")
    print(f"  Presidio cascade: {total_presidio_full*1000:7.1f}ms total, {total_presidio_full/len(ENGLISH_SAMPLES)*1000:.2f}ms/sample")
    print(f"  Our cascade:      {total_our_full*1000:7.1f}ms total, {total_our_full/len(ENGLISH_SAMPLES)*1000:.2f}ms/sample")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
