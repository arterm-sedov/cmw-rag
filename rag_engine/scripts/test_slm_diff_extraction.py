"""
SLM-based PII Extraction Test
Tests two approaches:
1. Rewrite + Diff (brittle, shown to fail)
2. Structured JSON Output (robust - recommended)
"""

import os
import json
import re
import time
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DATASET_PATH = r"D:\Repo\cmw-rag\rag_engine\data\synthetic_cmw_support_tickets_v1_with_pii.json"

# Try different models - use environment variable or fall back
MODEL_NAME = os.getenv("SLM_MODEL", "openai/gpt-oss-20b")

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT_JSON = """You are a PII extractor for IT SUPPORT TICKETS.

Your job is to extract ALL sensitive information from technical support requests.

## Entity Types (25 classes)

Extract these EXACT types:
1. NAME - Person names (Иван Иванов, John Smith, Петров В.А., Sarah Wilson)
2. COMPANY - Organizations (Газпромнефть, Microsoft, Сбербанк, Acme Corp)
3. EMAIL - Email addresses (user@example.com, ivanov@company.ru)
4. PHONE - Phone numbers in ANY format
5. INN - Russian Tax ID (10-12 digits)
6. IP_ADDRESS - IP addresses (IPv4, IPv6)
7. URL - Web links (https://support.example.com/ticket/123)
8. LOGIN - Usernames, accounts (admin, systemAccount, service_account)
9. PASSWORD - Passwords visible in text
10. API_KEY - API keys, tokens, secrets
11. ADDRESS - Full addresses
12. CITY - City names
13. BANK_CARD - Credit card numbers
14. BIC - Bank BIC/SWIFT codes
15. CAR_NUMBER - Vehicle plate numbers
16. PASSPORT - Passport numbers
17. SNILS - Russian SNILS
18. OGRN - Russian OGRN number
19. DATE - Dates
20. VERSION - Software versions (5.0.13334.0, v2.1.0)
21. US_SSN - US Social Security Number (XXX-XX-XXXX)
22. US_ZIP - US ZIP Code (12345, 12345-6789)
23. US_DRIVER_LICENSE - US Driver License
24. US_EIN - US Employer ID (XX-XXXXXXX)
25. US_NPI - US National Provider Identifier

## Use REGEX to find patterns

You're good at coding - use regex to find these patterns:

```
# Phone: Russian
\\+7\\s*\\(?\\d{3}\\)?\\s*\\d{3}[-\\s]?\\d{2}[-\\s]?\\d{2}
8\\s*\\(?\\d{3}\\)?\\s*\\d{3}[-\\s]?\\d{2}[-\\s]?\\d{2}

# Phone: US/International
\\+?1?[-.\\s]?\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}

# Email
[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}

# IP Address
\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b

# Russian INN (10-12 digits, context: инн, inn)
\\b\\d{10}\\b|\\b\\d{12}\\b

# Russian Passport (series + number)
\\b\\d{4}\\s*\\d{6}\\b

# US SSN
\\b\\d{3}-\\d{2}-\\d{4}\\b

# Bank Card (16 digits)
\\b\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}\\b

# URL
https?://[^\\s/$.?#].[^\\s]*

# Version numbers
\\b\\d+\\.\\d+(\\.\\d+)?\\b
```

## Examples (RU + EN, IT Support Context)

Example 1 (RU):
Input: "Добрый день! Пользователь Иван Иванов из компании Газпромнефть сообщил о проблеме. Email: ivanov@gazpromneft.ru, телефон: +7 495 123-45-67 доб. 100. Сервер 192.168.1.10 недоступен. Версия платформы: 5.0.13334.0. ИНН организации: 7714012345."
Output: [
  {"text": "Иван Иванов", "type": "NAME"},
  {"text": "Газпромнефть", "type": "COMPANY"},
  {"text": "ivanov@gazpromneft.ru", "type": "EMAIL"},
  {"text": "+7 495 123-45-67 доб. 100", "type": "PHONE"},
  {"text": "192.168.1.10", "type": "IP_ADDRESS"},
  {"text": "5.0.13334.0", "type": "VERSION"},
  {"text": "7714012345", "type": "INN"}
]

Example 2 (EN):
Input: "Contact John Smith at john.smith@acme.com. Server 10.0.0.50 has issues. US SSN: 123-45-6789. ZIP: 90210. See https://support.example.com/ticket/12345. Version 2.1.0"
Output: [
  {"text": "John Smith", "type": "NAME"},
  {"text": "john.smith@acme.com", "type": "EMAIL"},
  {"text": "10.0.0.50", "type": "IP_ADDRESS"},
  {"text": "123-45-6789", "type": "US_SSN"},
  {"text": "90210", "type": "US_ZIP"},
  {"text": "https://support.example.com/ticket/12345", "type": "URL"},
  {"text": "2.1.0", "type": "VERSION"}
]

Example 3 (Mixed):
Input: "Account systemAccount is locked. INN: 1234567890. Bank: 30101810400000000225. Card: 4111 1111 1111 1111."
Output: [
  {"text": "systemAccount", "type": "LOGIN"},
  {"text": "1234567890", "type": "INN"},
  {"text": "30101810400000000225", "type": "BANK_ACCOUNT"},
  {"text": "4111 1111 1111 1111", "type": "BANK_CARD"}
]

Example 4 (RU Company docs):
Input: "ОГРН: 1167746071421, КПП: 771401001, БИК: 044525201. Паспорт: 1234 567890."
Output: [
  {"text": "1167746071421", "type": "OGRN"},
  {"text": "771401001", "type": "KPP"},
  {"text": "044525201", "type": "BIC"},
  {"text": "1234 567890", "type": "PASSPORT"}
]

Example 5 (US Medical):
Input: "Provider NPI: 1234567890. EIN: 12-3456789. Driver license: A1234567. Address: 123 Main St, New York, NY 10001."
Output: [
  {"text": "1234567890", "type": "US_NPI"},
  {"text": "12-3456789", "type": "US_EIN"},
  {"text": "A1234567", "type": "US_DRIVER_LICENSE"},
  {"text": "123 Main St, New York, NY 10001", "type": "ADDRESS"}
]

## Output Format
Return ONLY a valid JSON array. No markdown, no explanation.
Extract EVERY entity you can find. Be aggressive! Use regex to find patterns.

Input: """


def anonymize_structured(text: str) -> list[dict]:
    """Use Structured JSON Output approach - much more robust than diff."""
    raw = ""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_JSON},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        
        # Debug: print first 100 chars of raw response
        print(f"  RAW: {raw[:100]}...")
        
        # Clean markdown code blocks if present
        if raw.startswith("```"):
            raw = re.sub(r'^```json?', '', raw)
            raw = re.sub(r'```$', '', raw)
        
        entities = json.loads(raw)
        return entities if isinstance(entities, list) else []
    except json.JSONDecodeError as e:
        print(f"JSON ERROR: {e}")
        print(f"RAW: {raw[:200] if raw else 'empty'}...")
        return []
    except Exception as e:
        print(f"ERROR: {e}")
        return []


def map_entities_to_placeholder(text: str, entities: list[dict]) -> tuple[str, dict]:
    """
    Map extracted entities back to source text using exact string matching.
    Returns (anonymized_text, mapping).
    """
    mapping = {}
    anonymized = text
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_entities = sorted(entities, key=lambda x: len(x['text']), reverse=True)
    
    for i, ent in enumerate(sorted_entities):
        entity_text = ent['text']
        entity_type = ent['type']
        
        # Normalize type NAME -> PERSON: for placeholder
        placeholder_type = "PERSON" if entity_type == "NAME" else entity_type
        
        # Find in original text (case-sensitive)
        if entity_text in anonymized:
            placeholder = f"[{placeholder_type}_{i+1}]"
            mapping[placeholder] = entity_text
            anonymized = anonymized.replace(entity_text, placeholder)
    
    return anonymized, mapping


def anonymize_with_slm_rewrite(text: str) -> str:
    """Legacy: Rewrite approach - less reliable."""
    SYSTEM_PROMPT = """You are a strict data anonymizer.
Rewrite the text by replacing names with [PERSON N] and companies with [COMPANY N].
Keep ALL other text IDENTICAL. Do not change punctuation, spacing, or casing."""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# Ground truth for testing
GROUND_TRUTH = [
    {"text": "Иван Иванов", "type": "NAME"},
    {"text": "Газпромнефть", "type": "COMPANY"},
    {"text": "Алексей Смирнов", "type": "NAME"},
    {"text": "Benjamin Keck", "type": "NAME"},
    {"text": "HC Companies", "type": "COMPANY"},
    {"text": "Jessica Casterline", "type": "NAME"},
    {"text": "Ramon Alarcon", "type": "NAME"},
    {"text": "Growscape", "type": "COMPANY"},
    {"text": "Microsoft", "type": "COMPANY"},
    {"text": "ООО ТехноСтрой", "type": "COMPANY"},
    {"text": "Петров В.А.", "type": "NAME"},
    {"text": "systemAccount", "type": "NAME"},  # Controversial - is it PII?
    {"text": "Kafka", "type": "COMPANY"},  # Should NOT be anonymized
]


def test_structured_output():
    print("=" * 70)
    print("TESTING: SLM Structured JSON Output Extraction")
    print("=" * 70)
    
    test_cases = [
        "Добрый день, я Иван Иванов из Газпромнефти. Прошу помочь с настройкой. Мой коллега, Алексей Смирнов, тоже не может войти.",
        "Hello, this is Benjamin Keck from HC Companies. Our IT admin Jessica Casterline reported an issue.",
        "Please contact Ramon Alarcon (RAlarcon@Growscape.com) regarding the Microsoft SQL database migration.",
        "Мы временно включали отправку от имени systemAccount, и теперь в кластере Kafka накопилось много технического мусора.",
        "У нас в ООО ТехноСтрой развернуто два приложения. Внедрением занимается Петров В.А.",
    ]
    
    for i, text in enumerate(test_cases):
        print(f"\n--- Sample {i + 1} ---")
        print(f"INPUT:  {text[:80]}...")
        
        # Get entities from SLM
        entities = anonymize_structured(text)
        print(f"SLM EXTRACTED: {entities}")
        
        # Map back to source
        anon_text, mapping = map_entities_to_placeholder(text, entities)
        print(f"ANONYMIZED:   {anon_text[:80]}...")
        print(f"MAPPING:      {mapping}")
        
        # Compare with ground truth
        gt_types = [e['type'] for e in GROUND_TRUTH if e['text'] in text]
        detected_types = [e['type'] for e in entities]
        print(f"GT TYPES:     {gt_types}")
        print(f"DETECTED:     {detected_types}")


def benchmark_on_dataset(n_samples: int = 20):
    """Run benchmark on synthetic dataset."""
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: Running on {n_samples} samples from enriched dataset")
    print("=" * 70)
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_entities_found = 0
    total_gt_entities = 0
    total_time = 0
    
    # For precision/recall calculation
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for i, item in enumerate(data[:n_samples]):
        text = item.get('question', '')
        ground_truth = item.get('ground_truth', [])
        
        if len(text) < 20:
            continue
            
        start = time.time()
        entities = anonymize_structured(text)
        elapsed = time.time() - start
        
        total_time += elapsed
        
        # Calculate metrics
        gt_texts = {e['text'] for e in ground_truth}
        found_texts = {e['text'] for e in entities}
        
        true_pos = gt_texts & found_texts
        false_pos = found_texts - gt_texts
        false_neg = gt_texts - found_texts
        
        true_positives += len(true_pos)
        false_positives += len(false_pos)
        false_negatives += len(false_neg)
        
        total_entities_found += len(entities)
        total_gt_entities += len(ground_truth)
        
        if i < 10:
            print(f"\nSample {i+1}: {len(entities)} found, {len(ground_truth)} GT in {elapsed:.2f}s")
            print(f"  Text: {text[:60]}...")
            print(f"  GT:      {[e['type'] for e in ground_truth]}")
            print(f"  Found:   {[e['type'] for e in entities]}")
            if false_pos:
                print(f"  FP: {false_pos}")
            if false_neg:
                print(f"  FN: {false_neg}")
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"RESULTS:")
    print(f"  Total samples: {n_samples}")
    print(f"  GT entities: {total_gt_entities}")
    print(f"  Found entities: {total_entities_found}")
    print(f"  Avg time/sample: {total_time/max(n_samples,1):.2f}s")
    print(f"")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    print("=" * 70)


def main():
    print(f"Using model: {MODEL_NAME}")
    
    # Test structured output approach
    test_structured_output()
    
    # Skip benchmark for now - too slow
    # benchmark_on_dataset(n_samples=10)


if __name__ == "__main__":
    main()
