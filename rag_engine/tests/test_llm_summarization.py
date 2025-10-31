from __future__ import annotations

from rag_engine.llm.summarization import summarize_to_tokens


class _FakeModel:
    def __init__(self, response: str):
        self._response = response

    def invoke(self, messages):  # noqa: ANN001
        return type("Resp", (), {"content": self._response})()


class _FakeLLM:
    def __init__(self, window: int, response: str):
        self._window = window
        self._model = _FakeModel(response)

    def get_current_llm_context_window(self) -> int:
        return self._window

    def _chat_model(self):  # noqa: ANN001
        return self._model


def test_summarize_to_tokens_uses_chunks_only_when_body_would_overflow():
    # Force tiny window so that adding full body would exceed
    llm = _FakeLLM(window=200, response="summary")
    title = "T"
    url = "http://u"
    chunks = ["relevant chunk"]
    full_body = "x" * 2000

    out = summarize_to_tokens(
        title=title,
        url=url,
        matched_chunks=chunks,
        full_body=full_body,
        target_tokens=100,
        guidance="Q",
        llm=llm,
        max_retries=0,
    )
    assert out == f"# {title}\n\nURL: {url}\n\n" + "summary"


def test_summarize_to_tokens_falls_back_to_stitched_on_empty_output():
    llm = _FakeLLM(window=100000, response="")
    title = "T2"
    url = "http://u2"
    chunks = ["c1", "c2"]
    out = summarize_to_tokens(
        title=title,
        url=url,
        matched_chunks=chunks,
        full_body=None,
        target_tokens=50,
        guidance="Q",
        llm=llm,
        max_retries=1,
    )
    assert out.startswith(f"# {title}\n\nURL: {url}")
    # Stitched fallback should contain both chunks
    assert "c1" in out and "c2" in out


