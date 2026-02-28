#!/usr/bin/env python
"""Create test records in TPAIModel template."""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from rag_engine.cmw_platform import records

# Create test records in TPAIModel
test_records = [
    (
        "Как установить платформу Comindware?",
        "Хочу установить Comindware Platform на свой сервер. Какие системные требования? Какой порядок установки? Какая версия самая последняя?",
    ),
    (
        "Как удалить платформу?",
        "Нужно удалить Comindware Platform с сервера. Какая процедура удаления? Нужно ли делать бэкап?",
    ),
    (
        "Какая последняя версия ПО?",
        "Какая актуальная версия Comindware Platform? Где посмотреть историю версий и release notes?",
    ),
    (
        "Какая лицензионная политика?",
        "Расскажите про лицензирование Comindware Platform. Какие типы лицензий доступны? Ограничения по пользователям?",
    ),
]

print("Creating test records in TPAIModel...")
created_ids = []

for title, question in test_records:
    result = records.create_record(
        application_alias="dima",
        template_alias="TPAIModel",
        values={
            "title": title,
            "user_question": question,
            "version": "5.0",
            "browser": "Chrome",
        },
    )
    if result["success"]:
        print(f"✓ Created: {result['record_id']} - {title}")
        created_ids.append(result["record_id"])
    else:
        print(f"✗ Failed: {title}")
        print(f"  Error: {result.get('error')}")

print()
print(f"Created {len(created_ids)} records: {created_ids}")
