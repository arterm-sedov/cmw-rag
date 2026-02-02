# ChromaDB HTTP Migration Plan

## 🎯 **Executive Summary**

**Yes, running ChromaDB as a separate process is the optimal solution** for your use case. This will:

- ✅ **Eliminate the 21-second initialization delay** (separate service startup)
- ✅ **Enable true async processing** (no blocking Gradio UI)
- ✅ **Provide connection pooling** (built-in HTTP client pooling)
- ✅ **Allow horizontal scaling** (multiple app instances → one ChromaDB service)
- ✅ **Improve resource isolation** (vector operations don't block main app)

## 📊 **Performance Analysis: HTTP vs PersistentClient**

| **Metric** | **PersistentClient (Current)** | **HTTP Client (Proposed)** |
|------------|--------------------------------|----------------------------|
| **Initial Connection** | 21+ seconds (file system + HNSW load) | ~50-200ms (TCP handshake) |
| **Query Latency** | 2-5ms (local SQLite) | 3-8ms (network + server processing) |
| **Memory Usage** | 3GB+ in app process | ~600MB in separate container |
| **Concurrent Users** | Limited by single process | Scales with HTTP server |
| **Failure Isolation** | App crash = vector store loss | Independent services |
| **Scaling** | Vertical only | Horizontal + Vertical |

## 🏗️ **Recommended Architecture**

```
┌─────────────────┐    HTTP API     ┌─────────────────────┐
│   Gradio App    │◄──────────────►│  ChromaDB Server    │
│  (async + UI)   │                │  (HTTP mode)        │
│                 │                │                     │
│  • Tool calls   │                │  • HNSW index       │
│  • Agent logic  │                │  • Connection pool  │
│  • Memory mgmt  │                │  • Persistent storage│
└─────────────────┘                └─────────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────┐                ┌─────────────────────┐
│  Thread Pool    │                │   Local Storage     │
│  (concurrency)  │                │   (./chroma_data)   │
└─────────────────┘                └─────────────────────┘
```

## 📝 **Implementation Plan**

### **Phase 0: Native ChromaDB Server** ⏱️ 10 minutes

**Start here for machines without Docker/sudo access.**

#### 0.1 Install and Run ChromaDB
```bash
# Install ChromaDB (same version as your app)
pip install chromadb

# Start ChromaDB HTTP server
chroma run --host localhost --port 8000 --path ./chroma_data

# Or with persistence path
chroma run --host 0.0.0.0 --port 8000 --path ./data/chromadb_data
```

#### 0.2 Environment Variables
```bash
# Add to .env
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
CHROMADB_SSL=false
CHROMADB_USE_HTTP=true
CHROMADB_CONNECTION_TIMEOUT=30.0
CHROMADB_MAX_CONNECTIONS=100
```

> **Note**: These env vars are read by `settings.py` using Pydantic's `Field(env="VAR_NAME")` for validation and centralized access.

#### 0.3 Verify Server
```bash
# Test connection
curl http://localhost:8000/api/v1/heartbeat

# Or use existing script (reads from .env automatically)
python rag_engine/scripts/test_chroma_connection.py
```

---

### **Phase 1: Docker ChromaDB Setup** (Optional) ⏱️ 30 minutes

**Use this when Docker is available for better process management and team consistency.**

#### 1.1 Docker Compose Configuration
```yaml
# docker-compose.yml (optional - for Docker environments)
version: '3.8'
services:
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data/chromadb_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
      - CHROMA_LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3
```

> **Ports**: Both `chroma run` and Docker use port 8000 by default. Environment variables remain identical.

#### 1.2 When to Use Docker vs Native

| **Scenario** | **Native (`chroma run`)** | **Docker** |
|--------------|---------------------------|------------|
| No sudo/Docker access | ✅ Yes | ❌ No |
| Quick local dev | ✅ Yes | ⚠️ Optional |
| Production deployment | ⚠️ Use systemd | ✅ Yes |
| Team consistency | ⚠️ Version pinning | ✅ Yes |
| Health monitoring | ⚠️ Manual | ✅ Built-in |
| Multiple services | ❌ Complex | ✅ Easy |

---

### **Phase 2: Code Migration** ⏱️ 2-3 hours

#### 2.1 Configuration Updates
```python
# rag_engine/config/settings.py - Add these settings
from pydantic import Field

chromadb_host: str = Field(default="localhost", env="CHROMADB_HOST")
chromadb_port: int = Field(default=8000, env="CHROMADB_PORT")
chromadb_ssl: bool = Field(default=False, env="CHROMADB_SSL")
chromadb_use_http: bool = Field(default=True, env="CHROMADB_USE_HTTP")
chromadb_connection_timeout: float = Field(default=30.0, env="CHROMADB_CONNECTION_TIMEOUT")
chromadb_max_connections: int = Field(default=100, env="CHROMADB_MAX_CONNECTIONS")
```

#### 2.2 Vector Store Refactoring
```python
# rag_engine/storage/vector_store.py - Key changes needed

class ChromaStore:
    def __init__(self, 
                 collection_name: str,
                 host: str = "localhost",
                 port: int = 8000,
                 use_async: bool = True,
                 **kwargs):
        self.host = host
        self.port = port
        self.use_async = use_async
        
    async def _get_async_client(self):
        if self._async_client is None:
            self._async_client = chromadb.AsyncHttpClient(
                host=self.host,
                port=self.port,
                ssl=self.ssl,
                settings=Settings(
                    chroma_server_ssl_verify=False,
                    chroma_server_connection_timeout=self.connection_timeout
                )
            )
        return self._async_client
    
    async def similarity_search_async(self, query_embedding: List[float], k: int = 5):
        client = await self._get_async_client()
        collection = await client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        results = await collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"]
        )
        return self._process_results(results)
```

> **Coverage**: All scripts using `ChromaStore` (build_index.py, maintain_chroma.py, search_kbid.py, etc.) will auto-update. Only `check_chroma.py` needs manual migration (uses direct chromadb client).

> **12-Factor Admin Scripts**: All admin/utility scripts must read configuration from environment variables (via `.env`), not hardcoded values or command-line arguments as primary config. Examples:
> - `test_chroma_connection.py` - reads `CHROMADB_HOST`, `CHROMADB_PORT` from `.env`
> - `check_chroma.py` - should read from `.env` instead of using `os.getenv()` with hardcoded defaults
> - `build_index.py`, `maintain_chroma.py` - already use `settings.py` which loads from `.env` ✓

#### 2.3 Retriever Integration
```python
# rag_engine/retrieval/retriever.py - Update to use async methods

class VectorRetriever:
    async def retrieve_async(self, query: str, k: int = 5):
        # Parallel multi-query execution
        query_tasks = [
            self.store.similarity_search_async(qv, k=k) 
            for qv in query_vectors
        ]
        results = await asyncio.gather(*query_tasks, return_exceptions=True)
        return self._merge_and_rerank(results)
```

### **Phase 3: Testing & Optimization** ⏱️ 1-2 hours

#### 3.1 Performance Testing Script
```python
# scripts/test_chromadb_performance.py
async def test_http_vs_persistent():
    queries = generate_test_queries(100)
    
    # Test HTTP client
    http_times = []
    for query in queries:
        start = time.time()
        await http_store.similarity_search_async(query, k=10)
        http_times.append(time.time() - start)
    
    print(f"HTTP Client - Mean: {np.mean(http_times):.3f}s, P95: {np.percentile(http_times, 95):.3f}s")
```

#### 3.2 Health Check Integration
```python
# Add to main app startup
async def check_chromadb_health():
    try:
        client = chromadb.HttpClient(host=settings.chromadb_host, port=settings.chromadb_port)
        client.heartbeat()  # Will raise exception if not healthy
        logger.info("✅ ChromaDB service is healthy")
    except Exception as e:
        logger.error(f"❌ ChromaDB health check failed: {e}")
        raise
```

## 🚀 **Deployment Strategy**

### **Step 1: Start ChromaDB Server**

Choose **ONE** of these approaches:

**Option A: Native (No Docker Required)**
```bash
# 1. Start ChromaDB server
chroma run --host localhost --port 8000 --path ./data/chromadb_data

# 2. In another terminal, verify
python rag_engine/scripts/test_chroma_connection.py
```

**Option B: Docker (If Available)**
```bash
# 1. Start ChromaDB container
docker-compose up -d chromadb

# 2. Verify health
docker-compose ps
```

### **Step 2: Update Application**
```bash
# 3. Update .env configuration (already done in Phase 0/1)
# CHROMADB_HOST=localhost
# CHROMADB_PORT=8000
# CHROMADB_USE_HTTP=true

# 4. Restart application (uses HTTP client now)
python rag_engine/api/app.py
```

### **Step 3: Production Deployment**

**For machines without Docker/sudo:**
```bash
# Use systemd service or process manager
# Example systemd service file:
# /etc/systemd/system/chromadb.service
[Unit]
Description=ChromaDB HTTP Server
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/app
ExecStart=/opt/app/.venv/bin/chroma run --host 0.0.0.0 --port 8000 --path ./data/chromadb_data
Restart=always

[Install]
WantedBy=multi-user.target
```

**For machines with Docker:**
```bash
# Use docker-compose as shown in Phase 1
docker-compose up -d chromadb
```

### **Step 4: Performance Monitoring**
```python
# Add these metrics to your logging
async def monitored_vector_search(query, k=5):
    start_time = time.time()
    try:
        results = await chromadb_store.similarity_search_async(query, k)
        duration = time.time() - start_time
        logger.info(f"ChromaDB query: {duration:.3f}s for k={k}")
        return results
    except Exception as e:
        logger.error(f"ChromaDB query failed after {time.time() - start_time:.3f}s: {e}")
        raise
```

## 📈 **Expected Performance Improvements**

### **Before (Current PersistentClient)**
```
User Query → [21s init] → 2-5ms search → Response
Multi-query → [21s init] → 4x(2-5ms) → Response
Concurrent → Blocked by single process
```

### **After (HTTP Client)**
```
User Query → 50-200ms connect → 3-8ms search → Response
Multi-query → 50-200ms connect → parallel 3-8ms each → Response  
Concurrent → Scales with HTTP server capacity
```

**Expected Results:**
- 🚀 **95% faster cold starts** (21s → 200ms)
- 🚀 **2-3x better concurrent performance**
- 📉 **50% memory usage in main app**
- 🔧 **Better error handling and monitoring**

## 🛠️ **Migration Risk Assessment**

| **Risk** | **Probability** | **Impact** | **Mitigation** |
|----------|----------------|-----------|----------------|
| Network connectivity issues | Low | Medium | Retry logic + health checks |
| Data migration issues | Low | High | Backup existing data + test migration |
| Performance regression | Low | Medium | Performance testing + rollback plan |
| Deployment environment constraints | Medium | Low | Native `chroma run` fallback |

## 💡 **Key Questions for Final Decision**

1. **Timeline**: Can you afford 2-3 hours of migration time now for immediate performance gains?

2. **Infrastructure**: Do you have Docker available in your deployment environment?
   - **Yes**: Use Phase 1 (Docker) for better management
   - **No**: Use Phase 0 (`chroma run`) - works everywhere

3. **Backup Strategy**: Should we keep the embedded ChromaDB as a fallback during initial testing?

4. **Monitoring**: Would you like detailed performance metrics during the migration phase?

## ✅ **My Recommendation**

**Proceed with Phase 0 immediately** (`chroma run`) - it works on all machines without Docker/sudo requirements. You can add Docker (Phase 1) later when available.

The combination of native HTTP server + async integration perfectly addresses your performance requirements and works on every deployment target.

---

## 📋 **Implementation Checklist**

### **Phase 0: Native Server (10 mins)** ⭐ **START HERE**
- [ ] Install ChromaDB: `pip install chromadb`
- [ ] Start server: `chroma run --host localhost --port 8000 --path ./chroma_data`
- [ ] Add environment variables to `.env`
- [ ] Test connectivity with `test_chroma_connection.py`
- [ ] Verify application connects successfully

### **Phase 1: Docker Setup (30 mins)** (Optional)
- [ ] Create `docker-compose.yml` with ChromaDB service
- [ ] Start Docker container and verify health
- [ ] Test basic connectivity from app
- [ ] Document for team/production use

### **Phase 2: Code Migration (2-3 hours)**
- [ ] Update `settings.py` with HTTP client configuration
- [ ] Refactor `vector_store.py` for async HTTP client
- [ ] Add async methods to `ChromaStore` class
- [ ] Update `retriever.py` to use async methods
- [ ] Implement health check in main app
- [ ] Add error handling and retry logic
- [ ] Update `check_chroma.py` to use HttpClient with env-based fallback

### **Phase 3: Testing & Optimization (1-2 hours)**
- [ ] Create performance testing script
- [ ] Run benchmarks comparing HTTP vs PersistentClient
- [ ] Implement performance monitoring
- [ ] Test concurrent load scenarios
- [ ] Verify data migration accuracy

### **Phase 4: Production Deployment**
- [ ] Backup existing ChromaDB data
- [ ] Deploy with native server OR Docker (environment-dependent)
- [ ] Monitor performance metrics
- [ ] Validate user experience improvements

---

## 📚 **References & Resources**

1. **ChromaDB Documentation**: https://docs.trychroma.com/
2. **ChromaDB HTTP Client**: https://cookbook.chromadb.dev/core/clients/
3. **Docker Compose Reference**: https://docs.docker.com/compose/
4. **Async Python Patterns**: https://docs.python.org/3/library/asyncio.html

---

*Last Updated: 2025-02-02*
*Author: OpenCode Agent*
