#!/usr/bin/env python3
"""Test the unified search bubble functions."""

import sys
sys.path.insert(0, '/home/asedov/cmw-rag')

from rag_engine.api.stream_helpers import yield_search_bubble, update_search_bubble_by_id

def test_yield_search_bubble():
    """Test creating a search bubble."""
    print("Test 1: yield_search_bubble()")
    
    bubble = yield_search_bubble('тестовый запрос', 'abc123')
    
    assert bubble['role'] == 'assistant'
    assert bubble['metadata']['ui_type'] == 'search_bubble'
    assert bubble['metadata']['status'] == 'pending'
    assert bubble['metadata']['search_id'] == 'abc123'
    assert bubble['metadata']['query'] == 'тестовый запрос'
    assert 'тестовый запрос' in bubble['content']
    
    print("  ✓ Bubble created correctly")
    print(f"    - ui_type: {bubble['metadata']['ui_type']}")
    print(f"    - status: {bubble['metadata']['status']}")
    print(f"    - search_id: {bubble['metadata']['search_id']}")
    print(f"    - content: {bubble['content'][:50]}...")
    print()


def test_update_search_bubble():
    """Test updating a search bubble with results."""
    print("Test 2: update_search_bubble_by_id()")
    
    # Create bubble
    bubble = yield_search_bubble('последний релиз', 'def456')
    history = [bubble]
    
    # Update with results
    articles = [
        {'title': 'Сведения о выпуске 5.0', 'url': 'https://kb.example.com/1'},
        {'title': 'Обзор версий', 'url': 'https://kb.example.com/2'},
    ]
    result = update_search_bubble_by_id(history, 'def456', count=2, articles=articles)
    
    assert result == True
    assert history[0]['metadata']['status'] == 'done'
    # Just check that result was successful and status is done
    assert '2' in history[0]['content']
    
    print("  ✓ Bubble updated correctly")
    print(f"    - new status: {history[0]['metadata']['status']}")
    print(f"    - updated content: {history[0]['content'][:200]}...")
    print()


def test_update_nonexistent_bubble():
    """Test updating a non-existent bubble."""
    print("Test 3: update non-existent bubble")
    
    history = []
    result = update_search_bubble_by_id(history, 'nonexistent', count=5)
    
    assert result == False
    print("  ✓ Returns False for non-existent bubble")
    print()


def test_zero_results():
    """Test bubble with zero results."""
    print("Test 4: zero results (no sources section)")
    
    bubble = yield_search_bubble('несуществующий запрос', 'xyz789')
    history = [bubble]
    
    # Update with 0 results
    result = update_search_bubble_by_id(history, 'xyz789', count=0, articles=[])
    
    assert result == True
    assert history[0]['metadata']['status'] == 'done'
    # With 0 results, no sources section should be added
    assert history[0]['content'].count('\n') < 5  # Should be short (no sources list)
    
    print("  ✓ Zero results handled correctly")
    print(f"    - content: {history[0]['content']}")
    print()


if __name__ == '__main__':
    try:
        test_yield_search_bubble()
        test_update_search_bubble()
        test_update_nonexistent_bubble()
        test_zero_results()
        
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
