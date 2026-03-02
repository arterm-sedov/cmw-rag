#!/usr/bin/env python3
"""
Generate a 100-sample synthetic dataset for PII anonymization evaluation.
Uses real patterns from 'Request for AI.csv' but replaces all sensitive data with synthetic values.
"""

import pandas as pd
import re
import json
import random
import os

# Configuration
SOURCE_CSV = r"D:\Documents\Request for AI.csv"
OUTPUT_JSON = r"D:\Repo\cmw-rag\.opencode\plans\anonymization_test_dataset.json"
TOTAL_TARGET = 200

random.seed(42)

# Synthetic Data Pools (Expanded for 200 samples)
SYNTHETIC_DATA = {
    'names_ru': [
        "Александр Иванов", "Елена Петрова", "Дмитрий Сидоров", "Наталья Козлова",
        "Сергей Волков", "Анна Смирнова", "Михаил Кузнецов", "Ольга Лебедева",
        "Игорь Соколов", "Татьяна Морозова", "Алексей Николаев", "Мария Иванова",
        "Владимир Фёдоров", "Светлана Дмитриева", "Павел Андреев", "Ирина Савельева",
        "Глеб Жигачев", "Артем Соколов", "Юлия Морозова", "Олег Павлов",
        "Николай Степанов", "Виктория Орлова", "Станислав Макаров", "Дарья Киселева",
        "Роман Зайцев", "Полина Борисова", "Вадим Никитин", "Марина Сорокина",
        "Артур Гусев", "Лилия Беляева", "Егор Воронин", "Алиса Щербакова"
    ],
    'names_en': [
        "John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis",
        "David Wilson", "Jennifer Martinez", "Christopher Lee", "Amanda White",
        "Robert Taylor", "Daniel Harris", "Lisa Anderson", "Mark Thompson",
        "Benjamin Keck", "Liesl Giorgio", "Jessica Casterline", "Michelle Racine",
        "Kevin King", "Nancy Lewis", "Paul Walker", "Ruth Young",
        "Steven Garcia", "Betty Robinson", "George Clark", "Karen Moore",
        "Jason Miller", "Patricia Taylor", "Thomas Moore", "Barbara Jackson",
        "Richard White", "Susan Harris", "Charles Martin", "Jessica Thompson"
    ],
    'orgs': [
        "ТехноСтрой", "Сбербанк", "Магнит", "Яндекс", "Газпром", "ВТБ", "Лукойл",
        "Роснефть", "СИБУР", "Северсталь", "Acme Corp", "Tech Solutions", "GlobalTech",
        "Innovation Inc", "CloudTech", "Enterprise Solutions", "ByteWave", "DataFlow Systems",
        "NextGen Tech", "CyberDyne Systems", "VelocityStack", "InnovaTech LLC", "CloudScale Inc",
        "NetFlow Systems", "HyperScale Tech", "PrimeCloud", "DevOps Pro", "AgileStack"
    ],
    'emails': [
        "alex.ivanov@example.ru", "elena.petrova@example.ru", "john.smith@acme.com",
        "sarah.j@techcorp.com", "user@company.ru", "support@company.net",
        "admin@enterprise.io", "info@globaltech.net", "dev@cloudscale.io",
        "contact@nextgen.tech", "billing@innovatech.co", "sales@bytewave.io"
    ],
    'phones_ru': [
        "+7-900-111-22-33", "+7-901-222-33-44", "+7-902-333-44-55", "+7-903-444-55-66",
        "+7-904-555-66-77", "+7-905-666-77-88", "+7-906-777-88-99", "8-900-123-45-67",
        "+7-910-111-22-33", "+7-911-222-33-44", "+7-912-333-44-55", "+7-913-444-55-66"
    ],
    'phones_us': [
        "+1-555-010-0001", "+1-555-010-0002", "555-123-4567", "555-234-5678",
        "440-539-5678", "800-225-7712", "555-901-2345", "212-555-0199",
        "310-555-0123", "415-555-0155", "702-555-0101", "305-555-0111"
    ],
    'ips': [
        "192.168.1.1", "192.168.0.100", "10.0.0.1", "10.0.1.50",
        "172.16.0.10", "192.168.10.25", "10.10.10.5", "192.168.22.22",
        "172.16.10.25", "10.20.0.100", "192.168.100.5", "10.100.0.25"
    ],
    'urls': [
        "https://example.com", "https://test.company.ru", "https://crm.example.net",
        "https://support.comindware.ru", "https://kb.example.org", "https://api.company.com",
        "https://prod.company.net", "https://dev.company.io"
    ]
}

def clean_html(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"<(p|br|div|li)[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def get_entities(text):
    entities = []
    patterns = {
        'EMAIL': r"[\w\.-]+@[\w\.-]+\.\w+",
        'PHONE': r"(\+7|8|(?:\+1-)?\d{3})[\s\-\(\)]?\d{3}[\s\-\(\)]?\d{2,4}[\s\-\(\)]?\d{2,4}",
        'IP_ADDRESS': r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        'URL': r"https?://[^\s<>\"]+",
        'DATE': r"\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2}"
    }
    for etype, pattern in patterns.items():
        for m in re.finditer(pattern, text):
            entities.append({'text': m.group(), 'type': etype, 'start': m.start(), 'end': m.end()})
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
    is_ru = cyrillic / max(len(text), 1) > 0.2
    if is_ru:
        for m in re.finditer(r"[А-Я][а-яё]+\s+[А-Я][а-яё]+(?:\s+[А-Я][а-яё]+)?", text):
            if m.group() not in ["Добрый день", "Коллеги", "Основание заявки", "Тип техподдержки", "Пункт перечня"]:
                entities.append({'text': m.group(), 'type': 'PERSON', 'start': m.start(), 'end': m.end()})
    else:
        for m in re.finditer(r"[A-Z][a-z]+\s+[A-Z][a-z]+", text):
            if m.group() not in ["Best Regards", "Chief Information", "Comindware Team"]:
                entities.append({'text': m.group(), 'type': 'PERSON', 'start': m.start(), 'end': m.end()})
    entities.sort(key=lambda x: (x['end'] - x['start']), reverse=True)
    final = []
    covered = [False] * len(text)
    for e in entities:
        if not any(covered[i] for i in range(e['start'], e['end'])):
            final.append(e)
            for i in range(e['start'], e['end']): covered[i] = True
    return sorted(final, key=lambda x: x['start'])

def synthesize_sample(text, entities):
    synth_text = text
    mapping = {}
    for e in entities:
        etype = e['type']
        if etype == 'EMAIL': val = random.choice(SYNTHETIC_DATA['emails'])
        elif etype == 'PHONE': 
            val = random.choice(SYNTHETIC_DATA['phones_ru'] if '\u0400' <= e['text'] <= '\u04FF' or '+7' in e['text'] or e['text'].startswith('8') else SYNTHETIC_DATA['phones_us'])
        elif etype == 'IP_ADDRESS': val = random.choice(SYNTHETIC_DATA['ips'])
        elif etype == 'URL': val = random.choice(SYNTHETIC_DATA['urls'])
        elif etype == 'PERSON':
            val = random.choice(SYNTHETIC_DATA['names_ru'] if any("\u0400" <= c <= "\u04FF" for c in e['text']) else SYNTHETIC_DATA['names_en'])
        elif etype == 'DATE': val = e['text']
        else: val = e['text']
        mapping[e['text']] = val
    for old, new in mapping.items():
        synth_text = synth_text.replace(old, new)
    final_entities = []
    for old, new in mapping.items():
        for m in re.finditer(re.escape(new), synth_text):
            etype = next(e['type'] for e in entities if e['text'] == old)
            final_entities.append({'text': new, 'type': etype, 'start': m.start(), 'end': m.end()})
    if not any(e['type'] == 'ORGANIZATION' for e in final_entities) and len(synth_text) > 100:
        org = random.choice(SYNTHETIC_DATA['orgs'])
        if "." in synth_text:
            pos = synth_text.find(".")
            synth_text = synth_text[:pos] + f" at {org}" + synth_text[pos:]
            final_entities.append({'text': org, 'type': 'ORGANIZATION', 'start': pos + 4, 'end': pos + 4 + len(org)})
    final_entities.sort(key=lambda x: x['start'])
    unique_entities = []
    last_end = -1
    for e in final_entities:
        if e['start'] >= last_end:
            unique_entities.append(e)
            last_end = e['end']
    return synth_text, unique_entities

def main():
    df = pd.read_csv(SOURCE_CSV, sep=";", encoding="utf-8-sig")
    with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
        existing = json.load(f)
    all_samples = existing['samples'][:50]
    needed = TOTAL_TARGET - len(all_samples)
    new_samples = []
    
    # First pass: try to get high-quality samples with entities
    for i in range(1, len(df)):
        if len(new_samples) >= needed: break
        clean_text = clean_html(df.iloc[i]['Описание'])
        if len(clean_text) < 40: continue
        entities = get_entities(clean_text)
        if not entities: continue
        synth_text, synth_entities = synthesize_sample(clean_text, entities)
        cyrillic = sum(1 for c in synth_text if "\u0400" <= c <= "\u04FF")
        lang = "ru" if cyrillic / max(len(synth_text), 1) > 0.2 else "en"
        new_samples.append({'id': 50 + len(new_samples) + 1, 'language': lang, 'text': synth_text[:600], 'entities': synth_entities})
    
    # Second pass: if still needed, take any text and force synthetic entities
    if len(new_samples) < needed:
        print(f"Only found {len(new_samples)} natural PII samples. Padding with forced synthetic samples...")
        for i in range(1, len(df)):
            if len(new_samples) >= needed: break
            clean_text = clean_html(df.iloc[i]['Описание'])
            if len(clean_text) < 100: continue # Need enough room for forced entities
            
            # Skip if already added in first pass
            if any(s['text'][:50] in clean_text for s in new_samples): continue
            
            # Force synthetic entities into plain text
            name = random.choice(SYNTHETIC_DATA['names_ru' if random.random() > 0.3 else 'names_en'])
            email = random.choice(SYNTHETIC_DATA['emails'])
            org = random.choice(SYNTHETIC_DATA['orgs'])
            
            prefix = f"Request from {name} ({email}) at {org}: "
            synth_text = prefix + clean_text
            
            synth_entities = [
                {'text': name, 'type': 'PERSON', 'start': 13, 'end': 13 + len(name)},
                {'text': email, 'type': 'EMAIL', 'start': 13 + len(name) + 2, 'end': 13 + len(name) + 2 + len(email)},
                {'text': org, 'type': 'ORGANIZATION', 'start': 13 + len(name) + 2 + len(email) + 5, 'end': 13 + len(name) + 2 + len(email) + 5 + len(org)}
            ]
            
            cyrillic = sum(1 for c in synth_text if "\u0400" <= c <= "\u04FF")
            lang = "ru" if cyrillic / max(len(synth_text), 1) > 0.2 else "en"
            new_samples.append({'id': 50 + len(new_samples) + 1, 'language': lang, 'text': synth_text[:600], 'entities': synth_entities})

    all_samples.extend(new_samples)
    existing['version'] = "2.1"
    existing['total_samples'] = len(all_samples)
    existing['samples'] = all_samples
    existing['language_mix'] = {'russian': sum(1 for s in all_samples if s['language'] == 'ru'), 'english': sum(1 for s in all_samples if s['language'] == 'en')}
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"Done! {len(all_samples)} samples. RU: {existing['language_mix']['russian']}, EN: {existing['language_mix']['english']}")

if __name__ == "__main__":
    main()
