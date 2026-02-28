"""Create a synthetic support request record in TPAIModel."""
from rag_engine.cmw_platform import records
import json

def create_synthetic_request():
    application = "dima"
    template = "TPAIModel"
    
    values = {
        "title": "Интеграция с 1С",
        "question": "<p>Подскажите, как настроить интеграцию Comindware Platform с 1С? Интересует возможность обмена данными о заказах. Версия платформы 5.0, используем Chrome.</p>",
        "version": "5.0",
        "browser": "Chrome"
    }
    
    print(f"Creating synthetic record in {application}.{template}...")
    result = records.create_record(application, template, values)
    
    if result["success"]:
        print(f"Success! Created record ID: {result['record_id']}")
        return result['record_id']
    else:
        print(f"Error: {result.get('error')}")
        print(f"Details: {json.dumps(result.get('data'), indent=2)}")
        return None

if __name__ == "__main__":
    create_synthetic_request()
