"""Enrich synthetic dataset with PII entities for benchmarking."""
import json
import random
import os
from pathlib import Path

# PII templates for enrichment
RUSSIAN_NAMES = [
    "Иван Иванов", "Алексей Смирнов", "Петров В.А.", "Екатерина Петрова",
    "Сергей Николаев", "Мария Кузнецова", "Дмитрий Волков", "Анна Морозова",
    "Андрей Фёдоров", "Ольга Соколова"
]

ENGLISH_NAMES = [
    "John Smith", "Benjamin Keck", "Jessica Casterline", "Ramon Alarcon",
    "Michael Brown", "Sarah Wilson", "David Lee", "Emily Johnson"
]

COMPANIES = [
    "Газпромнефть", "ООО ТехноСтрой", "Сбербанк", "Яндекс",
    "Microsoft", "Google", "HC Companies", "Growscape", "Acme Corp"
]

EMAILS = [
    "ivanov@gazpromneft.ru", "smirnov@company.ru", "petrov@techstroy.com",
    "john.smith@example.com", "ben.keck@hc-companies.com", "support@acme.com"
]

PHONES = [
    "+7 495 123-45-67", "8 800 555-35-35", "+1 555-123-4567",
    " доб. 1234", "+7 495 987-65-43"
]

INNS = ["1234567890", "7714012345", "5024056789"]
URLS = [
    "https://support.example.com/ticket/12345",
    "http://internal.corp.local/issue/56789",
    "https://jira.corp.local/browse/IT-1234"
]

IP_ADDRESSES = ["192.168.1.100", "10.0.0.50", "172.16.0.25"]

LOGINS = ["systemAccount", "admin", "service_account", "admin@corp"]

PASSWORDS = ["P@ssw0rd123", "SecretPass!", "MyP@ss2024!"]

ADDRESSES = [
    "г. Москва, ул. Ленина, д. 1",
    "Moscow, Tverskaya street, 1",
    "СПб, Невский пр., 25"
]


def enrich_sample(sample: dict) -> dict:
    """Add PII entities to a sample based on language and context."""
    lang = sample.get("language", "ru")
    question = sample.get("question", "")
    title = sample.get("title", "")
    text = f"{title}. {question}"
    
    # Determine PII to add based on context
    pii_entities = []
    
    # Always add at least 3-4 PII entities per sample
    num_entities = random.randint(3, 5)
    
    # Choose based on language
    if lang == "ru":
        names = RUSSIAN_NAMES
    else:
        names = ENGLISH_NAMES
    
    # Select unique entity types
    available_types = [
        ("NAME", names),
        ("COMPANY", COMPANIES),
        ("EMAIL", EMAILS),
        ("PHONE", PHONES),
        ("INN", INNS),
        ("URL", URLS),
        ("IP_ADDRESS", IP_ADDRESSES),
        ("LOGIN", LOGINS),
    ]
    
    # Randomly select entity types
    selected = random.sample(available_types, min(num_entities, len(available_types)))
    
    for etype, values in selected:
        entity_text = random.choice(values)
        pii_entities.append({"text": entity_text, "type": etype})
        # Inject entity into text
        text = _inject_entity(text, entity_text, etype, lang)
    
    sample['question'] = text
    sample['ground_truth'] = pii_entities
    
    return sample


def _inject_entity(text: str, entity: str, etype: str, lang: str) -> str:
    """Inject entity into text at a natural position."""
    templates = {
        "NAME": [
            f"Пользователь {entity} сообщил о проблеме." if lang == "ru" else f"User {entity} reported an issue.",
            f"Обратился {entity}." if lang == "ru" else f"Contact {entity} for details.",
            f"{entity} не может войти в систему." if lang == "ru" else f"{entity} cannot access the system.",
            f"Сообщение от {entity}." if lang == "ru" else f"Message from {entity}.",
        ],
        "COMPANY": [
            f"Организация {entity} использует нашу платформу." if lang == "ru" else f"Company {entity} uses our platform.",
            f"Клиент из {entity}." if lang == "ru" else f"Client from {entity}.",
            f"Интеграция с {entity}." if lang == "ru" else f"Integration with {entity}.",
        ],
        "EMAIL": [
            f"Напишите на {entity}." if lang == "ru" else f"Email {entity} for support.",
            f"Контакт: {entity}" if lang == "ru" else f"Contact: {entity}",
            f"Отправить на {entity}." if lang == "ru" else f"Send to {entity}.",
        ],
        "PHONE": [
            f"Звоните: {entity}" if lang == "ru" else f"Call: {entity}",
            f"Телефон для связи: {entity}" if lang == "ru" else f"Phone: {entity}",
            f"Контактный телефон: {entity}" if lang == "ru" else f"Contact phone: {entity}",
        ],
        "INN": [
            f"ИНН организации: {entity}" if lang == "ru" else f"INN: {entity}",
            f"ИНН: {entity}" if lang == "ru" else f"Tax ID: {entity}",
        ],
        "URL": [
            f"См. {entity}" if lang == "ru" else f"See {entity}",
            f"Ссылка: {entity}" if lang == "ru" else f"Link: {entity}",
            f"Подробнее: {entity}" if lang == "ru" else f"Details: {entity}",
        ],
        "IP_ADDRESS": [
            f"Сервер {entity} не отвечает." if lang == "ru" else f"Server {entity} is not responding.",
            f"Проблема на {entity}" if lang == "ru" else f"Issue at {entity}",
            f"Адрес: {entity}" if lang == "ru" else f"Address: {entity}",
        ],
        "LOGIN": [
            f"Учетная запись {entity} заблокирована." if lang == "ru" else f"Account {entity} is locked.",
            f"Вход под {entity} не работает." if lang == "ru" else f"Login {entity} doesn't work.",
            f"Пользователь {entity}." if lang == "ru" else f"User {entity}.",
        ],
    }
    
    template = random.choice(templates.get(etype, [f"{entity}"]))
    # Append to text
    return f"{text} {template}"


def main():
    # Load original dataset
    input_path = r"D:\Documents\synthetic_cmw_support_tickets_v1.json"
    output_path = Path(__file__).parent.parent / "data" / "synthetic_cmw_support_tickets_v1_with_pii.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Enrich each sample
    random.seed(42)  # For reproducibility
    enriched = [enrich_sample(sample) for sample in data]
    
    # Save enriched dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    
    # Print stats
    total_entities = sum(len(s.get('ground_truth', [])) for s in enriched)
    print(f"✅ Created enriched dataset: {output_path}")
    print(f"   Total samples: {len(enriched)}")
    print(f"   Total PII entities: {total_entities}")
    print(f"   Avg entities/sample: {total_entities/len(enriched):.1f}")
    
    # Show example
    print("\n--- Example ---")
    print(f"Title: {enriched[0]['title']}")
    print(f"Question: {enriched[0]['question'][:200]}...")
    print(f"Ground truth: {enriched[0]['ground_truth']}")


if __name__ == "__main__":
    main()
