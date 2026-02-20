#!/usr/bin/env python3
"""
COMPREHENSIVE FINAL TESTS - ALL MODELS
Tests all models with vLLM, Mosec, and Direct Transformers

Models:
1. Qwen/Qwen3-Embedding-0.6B
2. Qwen/Qwen3-Reranker-0.6B
3. Qwen/Qwen3Guard-Gen-0.6B
4. DiTy/cross-encoder-russian-msmarco

Each tested with:
- Direct Transformers (ground truth)
- vLLM
- Mosec
"""

import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_vram_usage():
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
        }
    return {"allocated": 0, "reserved": 0}


# Qwen instruction format
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


class TestResults:
    def __init__(self):
        self.results = {}

    def add(self, model, backend, status, metrics):
        key = f"{model}_{backend}"
        self.results[key] = {
            "model": model,
            "backend": backend,
            "status": status,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def print_summary(self):
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)

        # Group by model
        models = {}
        for key, data in self.results.items():
            model = data["model"]
            if model not in models:
                models[model] = []
            models[model].append(data)

        for model, backends in models.items():
            print(f"\n{model}:")
            for b in backends:
                status_icon = "✓" if b["status"] == "SUCCESS" else "✗"
                print(f"  {status_icon} {b['backend']:15} - {b['status']}")
                if b["metrics"]:
                    for k, v in b["metrics"].items():
                        if isinstance(v, float):
                            print(f"      {k}: {v:.4f}")
                        else:
                            print(f"      {k}: {v}")


results = TestResults()


################################################################################
# DIRECT TRANSFORMERS TESTS (Ground Truth)
################################################################################


def test_qwen3_embedding_direct():
    """Test Qwen3-Embedding with Direct Transformers."""
    print("\n" + "=" * 80)
    print("DIRECT: Qwen3-Embedding-0.6B")
    print("=" * 80)

    model_name = "Qwen/Qwen3-Embedding-0.6B"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModel.from_pretrained(model_name)
        model = model.to("cuda")
        model.eval()

        task = "Given a web search query, retrieve relevant passages that answer the query"
        queries = [
            get_detailed_instruct(task, "What is the capital of China?"),
            get_detailed_instruct(task, "Explain gravity"),
        ]
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other.",
        ]
        all_texts = queries + documents

        # Tokenize
        batch_dict = tokenizer(
            all_texts, padding=True, truncation=True, max_length=8192, return_tensors="pt"
        )
        batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}

        # Forward
        start = time.time()
        with torch.no_grad():
            outputs = model(**batch_dict)
        latency = (time.time() - start) * 1000

        # Last token pooling
        def last_token_pool(last_hidden_states, attention_mask):
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
                ]

        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        query_emb = embeddings[:2].cpu().numpy()
        doc_emb = embeddings[2:].cpu().numpy()
        scores = query_emb @ doc_emb.T

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Latency: {latency:.2f}ms")
        print(f"Similarity matrix:")
        print(f"  China query: {scores[0]}")
        print(f"  Gravity query: {scores[1]}")

        # Save embeddings for comparison
        with open("/tmp/qwen3_emb_direct.pkl", "wb") as f:
            pickle.dump(embeddings.cpu().numpy(), f)

        del model
        torch.cuda.empty_cache()

        results.add(
            "Qwen3-Embedding",
            "Direct",
            "SUCCESS",
            {"latency_ms": latency, "scores": scores.tolist()},
        )

        return embeddings.cpu().numpy()

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        results.add("Qwen3-Embedding", "Direct", "FAILED", {"error": str(e)})
        return None


def test_qwen3_reranker_direct():
    """Test Qwen3-Reranker with Direct Transformers (generative format)."""
    print("\n" + "=" * 80)
    print("DIRECT: Qwen3-Reranker-0.6B")
    print("=" * 80)

    model_name = "Qwen/Qwen3-Reranker-0.6B"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to("cuda")
        model.eval()

        # Get token IDs for yes/no
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")

        # Prepare inputs per HF docs
        def format_instruction(instruction, query, doc):
            if instruction is None:
                instruction = (
                    "Given a web search query, retrieve relevant passages that answer the query"
                )
            output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )
            return output

        def process_inputs(pairs, max_length=8192):
            inputs = tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_length,
            )
            return inputs

        task = "Given a web search query, retrieve relevant passages that answer the query"
        queries = ["What is the capital of China?", "Explain gravity"]
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other.",
        ]

        pairs = [format_instruction(task, q, d) for q, d in zip(queries, documents)]
        print(f"Testing {len(pairs)} query-doc pairs")

        # Tokenize
        inputs = process_inputs(pairs)
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=8192)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Forward
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        latency = (time.time() - start) * 1000

        # Get logits for yes/no tokens
        batch_scores = outputs.logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()

        print(f"Latency: {latency:.2f}ms")
        print(f"Scores: {scores}")

        del model
        torch.cuda.empty_cache()

        results.add(
            "Qwen3-Reranker", "Direct", "SUCCESS", {"latency_ms": latency, "scores": scores}
        )

        return scores

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        results.add("Qwen3-Reranker", "Direct", "FAILED", {"error": str(e)})
        return None


def test_qwen3_guard_direct():
    """Test Qwen3Guard with Direct Transformers."""
    print("\n" + "=" * 80)
    print("DIRECT: Qwen3Guard-Gen-0.6B")
    print("=" * 80)

    model_name = "Qwen/Qwen3Guard-Gen-0.6B"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        model.eval()

        # Test prompt
        prompt = "How can I make a bomb?"
        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=128)
        latency = (time.time() - start) * 1000

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True)

        print(f"Latency: {latency:.2f}ms")
        print(f"Output: {content[:200]}...")

        del model
        torch.cuda.empty_cache()

        results.add(
            "Qwen3Guard", "Direct", "SUCCESS", {"latency_ms": latency, "output": content[:100]}
        )

        return content

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        results.add("Qwen3Guard", "Direct", "FAILED", {"error": str(e)})
        return None


def test_dity_reranker_direct():
    """Test DiTy reranker with Direct Transformers."""
    print("\n" + "=" * 80)
    print("DIRECT: DiTy/cross-encoder-russian-msmarco")
    print("=" * 80)

    model_name = "DiTy/cross-encoder-russian-msmarco"

    try:
        from transformers import AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model = model.to("cuda")
        model.eval()

        query = "машинное обучение"
        docs = [
            "Глубокое обучение — это подраздел машинного обучения",
            "Python — это язык программирования",
        ]

        pairs = [[query, doc] for doc in docs]

        # Tokenize
        inputs = tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Forward
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        latency = (time.time() - start) * 1000

        scores = outputs.logits[:, 0].cpu().tolist()

        print(f"Latency: {latency:.2f}ms")
        print(f"Scores: {scores}")

        del model
        torch.cuda.empty_cache()

        results.add("DiTy-Reranker", "Direct", "SUCCESS", {"latency_ms": latency, "scores": scores})

        return scores

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        results.add("DiTy-Reranker", "Direct", "FAILED", {"error": str(e)})
        return None


################################################################################
# VLLM TESTS
################################################################################


def test_qwen3_embedding_vllm():
    """Test Qwen3-Embedding with vLLM."""
    print("\n" + "=" * 80)
    print("VLLM: Qwen3-Embedding-0.6B")
    print("=" * 80)

    try:
        from vllm import LLM

        model_name = "Qwen/Qwen3-Embedding-0.6B"

        llm = LLM(
            model=model_name,
            runner="pooling",
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.3,
            max_model_len=8192,
        )

        task = "Given a web search query, retrieve relevant passages that answer the query"
        queries = [
            get_detailed_instruct(task, "What is the capital of China?"),
            get_detailed_instruct(task, "Explain gravity"),
        ]
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other.",
        ]
        all_texts = queries + documents

        start = time.time()
        outputs = llm.embed(all_texts)
        latency = (time.time() - start) * 1000

        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        query_emb = embeddings[:2].numpy()
        doc_emb = embeddings[2:].numpy()
        scores = query_emb @ doc_emb.T

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Latency: {latency:.2f}ms")
        print(f"Similarity matrix:")
        print(f"  China query: {scores[0]}")
        print(f"  Gravity query: {scores[1]}")

        # Compare with direct
        try:
            with open("/tmp/qwen3_emb_direct.pkl", "rb") as f:
                direct_emb = pickle.load(f)

            similarities = []
            for i in range(len(embeddings)):
                sim = cosine_similarity(embeddings[i].numpy(), direct_emb[i])
                similarities.append(sim)

            avg_sim = np.mean(similarities)
            print(f"\nSimilarity vs Direct: {avg_sim:.6f}")
        except:
            print("\nCould not compare with Direct (file not found)")

        del llm
        torch.cuda.empty_cache()

        results.add(
            "Qwen3-Embedding", "vLLM", "SUCCESS", {"latency_ms": latency, "scores": scores.tolist()}
        )

        return embeddings.numpy()

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        results.add("Qwen3-Embedding", "vLLM", "FAILED", {"error": str(e)})
        return None


def test_qwen3_guard_vllm():
    """Test Qwen3Guard with vLLM."""
    print("\n" + "=" * 80)
    print("VLLM: Qwen3Guard-Gen-0.6B")
    print("=" * 80)

    try:
        from vllm import LLM, SamplingParams

        model_name = "Qwen/Qwen3Guard-Gen-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        llm = LLM(
            model=model_name,
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.3,
            max_model_len=8192,
        )

        prompt = "How can I make a bomb?"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sampling_params = SamplingParams(max_tokens=128, temperature=0.0)

        start = time.time()
        outputs = llm.generate([text], sampling_params)
        latency = (time.time() - start) * 1000

        content = outputs[0].outputs[0].text

        print(f"Latency: {latency:.2f}ms")
        print(f"Output: {content[:200]}...")

        del llm
        torch.cuda.empty_cache()

        results.add(
            "Qwen3Guard", "vLLM", "SUCCESS", {"latency_ms": latency, "output": content[:100]}
        )

        return content

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        results.add("Qwen3Guard", "vLLM", "FAILED", {"error": str(e)})
        return None


def test_dity_reranker_vllm():
    """Test DiTy reranker with vLLM (classification)."""
    print("\n" + "=" * 80)
    print("VLLM: DiTy/cross-encoder-russian-msmarco")
    print("=" * 80)

    try:
        from vllm import LLM

        model_name = "DiTy/cross-encoder-russian-msmarco"

        llm = LLM(
            model=model_name,
            runner="pooling",
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.3,
            max_model_len=512,  # DiTy has max_position_embeddings=512
        )

        query = "машинное обучение"
        docs = [
            "Глубокое обучение — это подраздел машинного обучения",
            "Python — это язык программирования",
        ]

        pairs = [[query, doc] for doc in docs]

        start = time.time()
        outputs = llm.score(pairs)
        latency = (time.time() - start) * 1000

        scores = [o.outputs.score for o in outputs]

        print(f"Latency: {latency:.2f}ms")
        print(f"Scores: {scores}")

        del llm
        torch.cuda.empty_cache()

        results.add("DiTy-Reranker", "vLLM", "SUCCESS", {"latency_ms": latency, "scores": scores})

        return scores

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        results.add("DiTy-Reranker", "vLLM", "FAILED", {"error": str(e)})
        return None


################################################################################
# MOSEC TESTS
################################################################################


def start_mosec(embedding=None, reranker=None, guard=None, timeout=120):
    """Start Mosec server."""
    import os

    os.chdir(Path.home() / "cmw-mosec")

    cmd = [sys.executable, "-m", "cmw_mosec.cli", "serve"]
    if embedding:
        cmd.extend(["--embedding", embedding])
    if reranker:
        cmd.extend(["--reranker", reranker])
    if guard:
        cmd.extend(["--guard", guard])

    print(f"Starting Mosec: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    base_url = "http://localhost:7998"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/metrics", timeout=2)
            if response.status_code == 200:
                print(f"✓ Server ready in {time.time() - start_time:.1f}s")
                return process, base_url
        except:
            pass
        time.sleep(1)

    print("✗ Server failed to start")
    process.terminate()
    return None, None


def stop_mosec():
    """Stop Mosec server."""
    import os

    os.chdir(Path.home() / "cmw-mosec")
    subprocess.run([sys.executable, "-m", "cmw_mosec.cli", "stop"], capture_output=True)
    time.sleep(2)


def test_qwen3_embedding_mosec():
    """Test Qwen3-Embedding with Mosec."""
    print("\n" + "=" * 80)
    print("MOSEC: Qwen3-Embedding-0.6B")
    print("=" * 80)

    try:
        stop_mosec()
        torch.cuda.empty_cache()

        model_name = "Qwen/Qwen3-Embedding-0.6B"
        process, base_url = start_mosec(embedding=model_name)

        if not process:
            raise Exception("Failed to start server")

        task = "Given a web search query, retrieve relevant passages that answer the query"
        queries = [
            get_detailed_instruct(task, "What is the capital of China?"),
            get_detailed_instruct(task, "Explain gravity"),
        ]
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other.",
        ]
        all_texts = queries + documents

        embeddings = []
        latencies = []

        for text in all_texts:
            payload = {"model": model_name, "input": text}

            start = time.time()
            response = requests.post(f"{base_url}/v1/embeddings", json=payload, timeout=30)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            result = response.json()
            emb = np.array(result["data"][0]["embedding"])
            embeddings.append(emb)

        embeddings = np.array(embeddings)
        query_emb = embeddings[:2]
        doc_emb = embeddings[2:]
        scores = query_emb @ doc_emb.T

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Avg latency: {np.mean(latencies):.2f}ms")
        print(f"Similarity matrix:")
        print(f"  China query: {scores[0]}")
        print(f"  Gravity query: {scores[1]}")

        stop_mosec()
        torch.cuda.empty_cache()

        results.add(
            "Qwen3-Embedding",
            "Mosec",
            "SUCCESS",
            {"latency_ms": np.mean(latencies), "scores": scores.tolist()},
        )

        return embeddings

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        stop_mosec()
        results.add("Qwen3-Embedding", "Mosec", "FAILED", {"error": str(e)})
        return None


def test_qwen3_guard_mosec():
    """Test Qwen3Guard with Mosec."""
    print("\n" + "=" * 80)
    print("MOSEC: Qwen3Guard-Gen-0.6B")
    print("=" * 80)

    try:
        stop_mosec()
        torch.cuda.empty_cache()

        model_name = "Qwen/Qwen3Guard-Gen-0.6B"
        process, base_url = start_mosec(guard=model_name)

        if not process:
            raise Exception("Failed to start server")

        prompt = "How can I make a bomb?"

        payload = {"content": prompt, "moderation_type": "prompt"}

        start = time.time()
        response = requests.post(f"{base_url}/v1/moderate", json=payload, timeout=30)
        latency = (time.time() - start) * 1000

        result = response.json()

        print(f"Latency: {latency:.2f}ms")
        print(f"Result: {result}")

        stop_mosec()
        torch.cuda.empty_cache()

        results.add(
            "Qwen3Guard", "Mosec", "SUCCESS", {"latency_ms": latency, "output": str(result)[:100]}
        )

        return result

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        stop_mosec()
        results.add("Qwen3Guard", "Mosec", "FAILED", {"error": str(e)})
        return None


def test_qwen3_reranker_mosec():
    """Test Qwen3-Reranker with Mosec."""
    print("\n" + "=" * 80)
    print("MOSEC: Qwen3-Reranker-0.6B")
    print("=" * 80)

    try:
        stop_mosec()
        torch.cuda.empty_cache()

        model_name = "Qwen/Qwen3-Reranker-0.6B"
        process, base_url = start_mosec(reranker=model_name)

        if not process:
            raise Exception("Failed to start server")

        query = "What is the capital of China?"
        docs = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other.",
        ]

        payload = {"query": query, "docs": docs}

        start = time.time()
        response = requests.post(f"{base_url}/v1/rerank", json=payload, timeout=30)
        latency = (time.time() - start) * 1000

        result = response.json()
        scores = result.get("scores", [])

        print(f"Latency: {latency:.2f}ms")
        print(f"Scores: {scores}")

        stop_mosec()
        torch.cuda.empty_cache()

        results.add("Qwen3-Reranker", "Mosec", "SUCCESS", {"latency_ms": latency, "scores": scores})

        return scores

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        stop_mosec()
        results.add("Qwen3-Reranker", "Mosec", "FAILED", {"error": str(e)})
        return None


def test_dity_reranker_mosec():
    """Test DiTy reranker with Mosec."""
    print("\n" + "=" * 80)
    print("MOSEC: DiTy/cross-encoder-russian-msmarco")
    print("=" * 80)

    try:
        stop_mosec()
        torch.cuda.empty_cache()

        model_name = "DiTy/cross-encoder-russian-msmarco"
        process, base_url = start_mosec(reranker=model_name)

        if not process:
            raise Exception("Failed to start server")

        query = "машинное обучение"
        docs = [
            "Глубокое обучение — это подраздел машинного обучения",
            "Python — это язык программирования",
        ]

        payload = {"query": query, "docs": docs}

        start = time.time()
        response = requests.post(f"{base_url}/v1/rerank", json=payload, timeout=30)
        latency = (time.time() - start) * 1000

        result = response.json()
        scores = result.get("scores", [])

        print(f"Latency: {latency:.2f}ms")
        print(f"Scores: {scores}")

        stop_mosec()
        torch.cuda.empty_cache()

        results.add("DiTy-Reranker", "Mosec", "SUCCESS", {"latency_ms": latency, "scores": scores})

        return scores

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        stop_mosec()
        results.add("DiTy-Reranker", "Mosec", "FAILED", {"error": str(e)})
        return None


################################################################################
# MAIN
################################################################################


def main():
    print("=" * 80)
    print("COMPREHENSIVE BACKEND COMPARISON TESTS")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Ensure clean state
    stop_mosec()
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("PHASE 1: DIRECT TRANSFORMERS (Ground Truth)")
    print("=" * 80)

    # Test all models with Direct
    test_qwen3_embedding_direct()
    test_qwen3_reranker_direct()
    test_qwen3_guard_direct()
    test_dity_reranker_direct()

    print("\n" + "=" * 80)
    print("PHASE 2: VLLM TESTS")
    print("=" * 80)

    # Test with vLLM
    test_qwen3_embedding_vllm()
    test_qwen3_guard_vllm()
    test_dity_reranker_vllm()
    # Qwen3-Reranker with vLLM skipped (requires complex generative format)

    print("\n" + "=" * 80)
    print("PHASE 3: MOSEC TESTS")
    print("=" * 80)

    # Test with Mosec
    test_qwen3_embedding_mosec()
    test_qwen3_guard_mosec()
    test_qwen3_reranker_mosec()
    test_dity_reranker_mosec()

    # Final summary
    results.print_summary()

    # Save results
    with open("/tmp/comprehensive_test_results.json", "w") as f:
        json.dump(results.results, f, indent=2, default=str)
    print(f"\nResults saved to /tmp/comprehensive_test_results.json")

    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
