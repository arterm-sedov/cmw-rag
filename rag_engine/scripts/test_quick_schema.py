"""
Quick test: GPT-4o vs Qwen3-8B with JSON Schema
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

TEXT = "Contact John Smith at john.smith@company.com. Server 10.0.0.50 has issues. INN: 1234567890."

PROMPT = """Extract PII entities from IT support tickets.

Entity types: NAME, COMPANY, EMAIL, PHONE, INN, IP_ADDRESS

Return JSON array: [{"text": "...", "type": "..."}]

Input: """

def test_json_schema(model: str):
    print(f"\n{'='*60}\nTesting {model} with JSON Schema\n{'='*60}")
    
    json_schema = {
        "name": "pii_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "type": {"type": "string"}
                        },
                        "required": ["text", "type"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["entities"],
            "additionalProperties": False
        }
    }
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": TEXT}
            ],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            }
        )
        
        raw = response.choices[0].message.content
        print(f"Raw response:\n{raw}\n")
        
        data = json.loads(raw)
        if "entities" in data:
            print("✅ Success! Entities:")
            for ent in data["entities"]:
                print(f"  - {ent['type']}: {ent['text']}")
        else:
            print("❌ No 'entities' key in response")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def test_tool_calling(model: str):
    print(f"\n{'='*60}\nTesting {model} with Tool Calling\n{'='*60}")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_pii",
                "description": "Extract PII entities. MANDATORY: Call this function.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "type": {"type": "string"}
                                },
                                "required": ["text", "type"]
                            }
                        }
                    },
                    "required": ["entities"]
                }
            }
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT + "\n\nMANDATORY: Call extract_pii function."},
                {"role": "user", "content": TEXT}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "extract_pii"}},
            temperature=0.0
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            print(f"Tool call arguments:\n{tool_call.function.arguments}\n")
            
            data = json.loads(tool_call.function.arguments)
            if "entities" in data:
                print("✅ Success! Entities:")
                for ent in data["entities"]:
                    print(f"  - {ent['type']}: {ent['text']}")
            else:
                print("❌ No 'entities' key in tool call")
        else:
            print("❌ No tool call returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Test both models
    test_json_schema("qwen/qwen3-8b")
    test_tool_calling("qwen/qwen3-8b")
    test_json_schema("openai/gpt-4o")
    test_tool_calling("openai/gpt-4o")
