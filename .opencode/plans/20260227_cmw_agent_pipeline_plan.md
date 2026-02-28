# CMW Platform Agent Pipeline Script Plan

## Goal

Create a new script `process_cmw_record.py` that implements the complete pipeline:
1. Fetch a record from CMW Platform using config from YAML
2. Build a structured markdown request from fetched attributes (config from YAML)
3. Call the RAG agent
4. Map agent response to CMW fields using config from YAML
5. Create a response record in the platform linked to the original

**Key principle:** All mapping and configuration lives in `cmw_platform.yaml` - the script reads this file to know what to fetch, how to build the request, and how to map the response.

---

## YAML Configuration (`cmw_platform.yaml`)

The YAML serves as the single source of truth. Combined structure - each output attribute includes both its type AND the agent source field.

```yaml
# CMW Platform Configuration
# Single source of truth for the agent pipeline

# Pipeline: input → request → output mapping
pipeline:
  # Input: template to fetch records from
  input:
    application: dima
    template: TPAIModel
    # Fields to fetch for building the request (name = template attribute)
    fields:
      - name: title
      - name: user_question
      - name: version
      - name: browser

  # Request builder: how to combine fields into markdown
  request_template: |
    # {title}

    {user_question}

    ---
    Metadata:
    - Version: {version}
    - Browser: {browser}

  # Output: template to create response in
  output:
    application: dima
    template: agent_responses
    # Link field to original record
    link_field: support_request
    link_to_template: TPAIModel

# Templates: combined attribute schema + agent mapping
templates:
  dima:
    # Input template (just schema, no mapping)
    TPAIModel:
      attributes:
        title: string
        user_question: string
        version: string
        browser: string

    # Output template: schema + from_agent mapping for each attribute
    agent_responses:
      attributes:
        # Link to original record (special - uses input record ID)
        support_request:
          type: record
          reference_to: TPAIModel
          from_agent: _input_record_id

        # Agent response mappings
        confidence:
          type: decimal
          from_agent: plan.confidence

        queries_count:
          type: integer
          from_agent: len(plan.knowledge_base_search_queries)

        queries:
          type: json
          from_agent: plan.knowledge_base_search_queries

        is_safe:
          type: boolean
          from_agent: safety.is_safe

        safety_categories:
          type: json
          from_agent: safety.categories

        user_intent:
          type: string
          from_agent: plan.user_intent

        topic:
          type: string
          from_agent: plan.topic

        category:
          type: string
          from_agent: plan.category

        intent_confidence:
          type: decimal
          from_agent: plan.intent_confidence

        spam_score:
          type: decimal
          from_agent: plan.spam_score

        spam_reason:
          type: string
          from_agent: plan.spam_reason

        action:
          type: string
          from_agent: plan.action

        engineer_intervention_needed:
          type: boolean
          from_agent: resolution.engineer_intervention_needed

        issue_summary:
          type: string
          from_agent: resolution.issue_summary

        steps_completed:
          type: json
          from_agent: resolution.steps_completed

        next_steps:
          type: json
          from_agent: resolution.next_steps

        outcome:
          type: string
          from_agent: resolution.outcome

        agent_answer:
          type: text
          from_agent: answer_text

        articles_found:
          type: integer
          from_agent: len(final_articles)
```

**Key improvements:**
- Each output attribute has `type` AND `from_agent` in one place
- No duplication between `templates:` and separate `field_mapping:`
- Input template attributes don't need `from_agent` (they come from CMW)
- `from_agent` supports:
  - Direct: `plan.user_intent`
  - Nested: `resolution.steps_completed`
  - Computed: `len(plan.queries)`

---

## Pipeline Script (`process_cmw_record.py`)

The script reads `cmw_platform.yaml` and executes the pipeline:

```python
import asyncio
import json
import yaml
from pathlib import Path

async def run_pipeline(record_id: str, dry_run: bool = False):
    # 1. Load config from YAML
    config = load_cmw_config()

    # 2. Fetch input record using config
    input_config = config["pipeline"]["input"]
    fields = [f["name"] for f in input_config["fields"]]
    record = read_record(record_id, fields=fields)

    # 3. Build request using template from YAML
    request_template = config["pipeline"]["request_template"]
    record_data = record["data"][record_id]
    md_request = request_template.format(**record_data)

    # 4. Call RAG agent
    from rag_engine.api import app as api_app
    result = await api_app.ask_comindware_structured(md_request)

    # 5. Map response using attributes with from_agent from YAML
    output_config = config["pipeline"]["output"]
    template_config = config["templates"][output_config["application"]][output_config["template"]]
    
    mapped_values = map_agent_response(
        agent_result=result,
        input_record_id=record_id,
        attributes=template_config["attributes"],
    )

    # 6. Create output record using config
    if not dry_run:
        create_record(
            application_alias=output_config["application"],
            template_alias=output_config["template"],
            values=mapped_values,
        )
```

---

## Implementation Details

### Config Loader Updates

`rag_engine/cmw_platform/config.py` gets new functions:
- `load_pipeline_config()` - loads pipeline section
- `get_input_config()` - returns input template/fields
- `get_output_config()` - returns output template config
- `get_output_attributes(app, template)` - returns attributes with from_agent mappings

### Response Mapper

`rag_engine/cmw_platform/mapping.py`:
- `map_agent_response(agent_result, input_record_id, attributes)` 
- Iterates over attributes, extracts values from agent_result using `from_agent`
- Handles:
  - Dot notation traversal: `plan.user_intent` → `result.plan.user_intent`
  - Functions: `len(items)` → `len(result.items)`
  - JSON serialization: lists/dicts → JSON strings
  - Special `_input_record_id` → links to original record

### Request Builder

`rag_engine/cmw_platform/request_builder.py`:
- `build_request(record_data, template)` - formats markdown from record data

---

## CLI Usage

```bash
# Process a single record
python rag_engine/scripts/process_cmw_record.py --record-id 322393

# Dry run (fetch, process, show output, don't create)
python rag_engine/scripts/process_cmw_record.py --record-id 322393 --dry-run
```

---

## File Changes

### New Files
- `rag_engine/cmw_platform/mapping.py` - Response mapping logic
- `rag_engine/cmw_platform/request_builder.py` - Markdown request builder
- `rag_engine/scripts/process_cmw_record.py` - Main pipeline script

### Modified Files
- `rag_engine/config/cmw_platform.yaml` - Simplified to combined structure

---

## Implementation Order

1. Update `cmw_platform.yaml` with simplified combined structure
2. Create `rag_engine/cmw_platform/mapping.py`
3. Create `rag_engine/cmw_platform/request_builder.py`
4. Create `rag_engine/scripts/process_cmw_record.py`
5. Add unit tests for mapping
6. Run integration test

---

## Validation

The script should:
- Successfully fetch record from TPAIModel using config
- Build correct markdown from fetched fields
- Call agent and get structured response
- Map all fields using `from_agent` mappings
- Create linked record in platform
