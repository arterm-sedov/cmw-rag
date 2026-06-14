# Systemd Service Plan — CMW RAG Stack

## Objective

Replace tmux-based process management with systemd user services for the three long-running processes that comprise the RAG stack: Mosec (embedding/reranker/guard), ChromaDB (vector store), and the Gradio RAG app. Also provide a reusable skill for starting/stopping/checking the stack.

## Architecture

```
                    ┌──────────────────┐
                    │  cmw-rag.target  │  ← groups all 3
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                 ▼
   ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
   │  cmw-rag-mosec  │ │ cmw-rag-chroma│ │  cmw-rag-app  │
   │  Type=simple    │ │ Type=simple   │ │ Type=simple   │
   │  port 7998      │ │ port 8000     │ │ port 7860     │
   │  Wants=network  │ │ Wants=network │ │ After=mosec   │
   │                 │ │               │ │ After=chroma  │
   └────────────────┘ └──────────────┘ └───────┬────────┘
                                               │
                                        (embeddings,
                                         reranking,
                                         guard)
```

## Service Files

All four files live under `systemd/` in the repo, installed via symlink or copy to `~/.config/systemd/user/`.

### 1. `systemd/cmw-rag-mosec.service`
- WorkingDirectory: `%h/cmw-mosec`
- ExecStart: `%h/cmw-mosec/.venv/bin/cmw-mosec serve`
- Restart: `on-failure`
- Type: `simple`
- SyslogIdentifier: `cmw-rag-mosec`
- After: `network-online.target`
- Wants: `network-online.target`

### 2. `systemd/cmw-rag-chroma.service`
- WorkingDirectory: `%h/cmw-rag`
- ExecStart: `%h/cmw-rag/.venv/bin/chroma run --host 0.0.0.0 --port 8000 --path %h/cmw-rag/data/chromadb_data`
- Restart: `on-failure`
- Type: `simple`
- SyslogIdentifier: `cmw-rag-chroma`
- After: `network-online.target`
- Wants: `network-online.target`

### 3. `systemd/cmw-rag-app.service`
- WorkingDirectory: `%h/cmw-rag`
- Environment: `PYTHONPATH=%h/cmw-rag`
- ExecStart: `%h/cmw-rag/.venv/bin/python rag_engine/api/app.py`
- Restart: `always`
- Type: `simple`
- SyslogIdentifier: `cmw-rag-app`
- After: `network-online.target cmw-rag-mosec.service cmw-rag-chroma.service`
- Wants: `cmw-rag-mosec.service cmw-rag-chroma.service`

### 4. `systemd/cmw-rag.target`
- Binds `cmw-rag-mosec.service`, `cmw-rag-chroma.service`, `cmw-rag-app.service`
- Allows `systemctl --user start/stop/status cmw-rag.target` to control the whole stack

## Installation Steps

```bash
# Link service files to systemd user directory
mkdir -p ~/.config/systemd/user
for f in systemd/cmw-rag-*.service systemd/cmw-rag.target; do
  ln -sf "$(pwd)/$f" ~/.config/systemd/user/
done

# Reload daemon and enable
systemctl --user daemon-reload
systemctl --user enable cmw-rag-mosec.service
systemctl --user enable cmw-rag-chroma.service
systemctl --user enable cmw-rag-app.service
systemctl --user enable cmw-rag.target

# Ensure lingering for user services (survive logout)
loginctl enable-linger $(whoami)
```

## Usage Commands

```bash
# Full stack
systemctl --user start cmw-rag.target
systemctl --user stop cmw-rag.target
systemctl --user status cmw-rag.target

# Individual services
systemctl --user start/stop/status cmw-rag-mosec
systemctl --user start/stop/status cmw-rag-chroma
systemctl --user start/stop/status cmw-rag-app

# Logs
journalctl --user -u cmw-rag-app -f
```

## Skill

Add `.agents/skills/cmw-rag-service-control/SKILL.md` that documents:
- When to use the skill (starting/stopping/checking the stack)
- All service commands
- Dependency order
- How to check logs per service
- Troubleshooting tips

## Files Changed

| File | Action |
|------|--------|
| `systemd/cmw-rag-mosec.service` | Create |
| `systemd/cmw-rag-chroma.service` | Create |
| `systemd/cmw-rag-app.service` | Create |
| `systemd/cmw-rag.target` | Create |
| `.agents/skills/cmw-rag-service-control/SKILL.md` | Create |

## Verification

```bash
# Check all services are running
systemctl --user status cmw-rag.target
systemctl --user status cmw-rag-mosec
systemctl --user status cmw-rag-chroma
systemctl --user status cmw-rag-app

# Check app responds
curl -s -o /dev/null -w "%{http_code}" http://localhost:7860

# Check logs for errors
journalctl --user -u cmw-rag-app --since "5 min ago" | grep -i error
journalctl --user -u cmw-rag-mosec --since "5 min ago" | grep -i error
journalctl --user -u cmw-rag-chroma --since "5 min ago" | grep -i error

# Cleanup: disable old tmux sessions if no longer needed
```

## Non-Goals

- No changes to Mosec configuration or cmw-mosec repo.
- No environment file handling (.env already loaded by the app).
- No log rotation configuration (journald default).
- No timers (sync service already exists separately).
- No changes to the existing sync service/timer.
