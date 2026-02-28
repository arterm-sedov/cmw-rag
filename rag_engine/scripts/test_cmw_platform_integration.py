from __future__ import annotations

from rag_engine.cmw_platform import create_record, read_record


def test_platform_integration():
    """Integration test against actual Comindware Platform."""
    
    # Test 1: Read record 322393 from TPAIModel template
    print("=" * 60)
    print("TEST 1: Read record 322393 from TPAIModel")
    print("=" * 60)
    
    read_result = read_record(
        record_id="322393",
        fields=["title", "user_question"]
    )
    
    print(f"Success: {read_result['success']}")
    print(f"Data: {read_result['data']}")
    print(f"Error: {read_result['error']}")
    print()
    
    if not read_result["success"]:
        print("FAILED: Could not read record")
        return
    
    # Test 2: Create new record in response template
    print("=" * 60)
    print("TEST 2: Create new record in response template")
    print("=" * 60)
    
    synthetic_resolution = (
        "Issue resolved after analyzing the reported problem. "
        "Customer reported difficulty accessing their account dashboard. "
        "Investigation revealed cache corruption in browser settings. "
        "Cleared browser cache and cookies, verified account credentials, "
        "and confirmed full system functionality restored."
    )
    
    # System names: app=dima, template=response
    create_result = create_record(
        application_alias="dima",
        template_alias="response",
        values={
            "request": "322393",
            "exampletext": synthetic_resolution,
        },
    )
    
    print(f"Success: {create_result['success']}")
    print(f"Status Code: {create_result['status_code']}")
    print(f"Record ID: {create_result['record_id']}")
    print(f"Error: {create_result['error']}")
    print()
    
    if create_result["success"] and create_result.get("record_id"):
        print(f"SUCCESS: Created new record with ID: {create_result['record_id']}")
    else:
        print(f"FAILED: {create_result.get('error', 'Unknown error')}")
        print(f"Full data: {create_result.get('data')}")


if __name__ == "__main__":
    test_platform_integration()
