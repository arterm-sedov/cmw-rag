from __future__ import annotations

import sys
from types import SimpleNamespace


def test_run_gradio_uvicorn_uses_bounded_graceful_shutdown(monkeypatch):
    from rag_engine.api import server

    calls = []

    def fake_run(asgi_app, **kwargs):
        calls.append((asgi_app, kwargs))

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))

    mounted_app = object()
    server.run_gradio_uvicorn(mounted_app)

    assert calls == [
        (
            mounted_app,
            {
                "host": server.settings.gradio_server_name,
                "port": server.settings.gradio_server_port,
                "timeout_graceful_shutdown": 3,
            },
        )
    ]
