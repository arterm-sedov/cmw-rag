# ChromaDB Host Variables Rename Plan

## Goal
Rename `CHROMADB_HOST` to separate server bind address from client connect address.

## New Variables

```bash
CHROMA_CLIENT_HOST=localhost   # Where RAG agent connects (client)
CHROMA_SERVER_BIND=0.0.0.0    # Where ChromaDB server listens (server)
```

## Non-Breaking Strategy
- Update ALL references to `CHROMADB_HOST` across codebase
- No deprecation warnings period - clean rename
- Update `.env` and `.env-example` together

## Files to Modify

### Configuration
| File | Change |
|------|--------|
| `rag_engine/config/settings.py` | Add `CHROMA_SERVER_BIND`, rename `chromadb_host` → `CHROMA_CLIENT_HOST` |
| `.env-example` | Replace `CHROMADB_HOST` with both new vars |
| `.env` | Replace `CHROMADB_HOST` with both new vars |

### Application
| File | Change |
|------|--------|
| `rag_engine/storage/vector_store.py` | `settings.chromadb_host` → `settings.chroma_client_host` |
| `rag_engine/api/app.py` | `settings.chromadb_host` → `settings.chroma_client_host` |

### Helper Scripts
| File | Change |
|------|--------|
| `rag_engine/scripts/start_chroma_server.py` | Read `CHROMA_SERVER_BIND` for `--host` arg |
| `rag_engine/scripts/test_chroma_connection.py` | Read `CHROMA_CLIENT_HOST` for connection |
| `rag_engine/scripts/check_chroma.py` | Read `CHROMA_CLIENT_HOST` for connection |

### Documentation
| File | Change |
|------|--------|
| `README.md` | Update `CHROMADB_HOST` references |
| `docs/ASYNC_IMPLEMENTATION_SUMMARY.md` | Update config references |

### Tests
| File | Change |
|------|--------|
| `rag_engine/tests/test_*.py` | Update any `CHROMADB_HOST` or `chromadb_host` references |

## Implementation Order

1. Update `settings.py` with new field names
2. Update `.env-example` with new variables + docs
3. Update `.env` with actual values
4. Update `storage/vector_store.py` and `api/app.py`
5. Update all helper scripts
6. Update documentation
7. Run tests to verify

## Verification

```bash
# Test connection works
python rag_engine/scripts/test_chroma_connection.py

# Run tests
pytest rag_engine/tests/ -v

# Start server
python rag_engine/scripts/start_chroma_server.py --foreground
```

## Backward Compatibility
**None** - this is a clean rename. All references updated in one PR.

## Estimated Effort
2-3 hours | 1 day

## Dependent Scripts
All scripts listed above rely on these variables:
- `start_chroma_server.py` - server bind
- `test_chroma_connection.py` - client connect
- `check_chroma.py` - client connect
- `maintain_chroma.py` - check if uses host var
