# OpenCode Troubleshooting Guide

Investigation findings and workarounds for OpenCode TUI hangs (Mar 2026).

## Symptoms

- OpenCode TUI (`opencode` with no args) hangs immediately — blank screen, requires Ctrl+C
- Works only when run directly from the **login shell** (`/bin/sh`), not from nested shells (bash, sh from bash, etc.)
- `opencode run "prompt"` (CLI mode) works in more contexts

## Root Cause Summary

| Context | Result |
|---------|--------|
| Login shell (sh) → `opencode` | ✅ Works |
| Login shell → bash → `opencode` | ❌ Hangs |
| Login shell → bash → sh → `opencode` | ❌ Hangs |
| `script -q -c "opencode"` | ❌ Hangs (TUI) |
| `script -q -c "opencode run hello"` | ✅ Works (CLI) |
| `ssh -t` | ❌ Still hangs if bash is used |

The TUI is sensitive to process hierarchy and shell context. Only the direct login shell provides a working environment.

## Workarounds

### 1. Stay in Login Shell (Recommended)

Do **not** run `bash` in the session where you need OpenCode. Use the default shell (`/bin/sh`) that appears right after SSH login.

```bash
# After SSH login, run opencode directly — do NOT run bash first
$ opencode
```

### 2. Use CLI Mode from Bash

When in bash, use the non-interactive CLI instead of the TUI:

```bash
opencode run "your question or task here"
```

### 3. Use Separate SSH Sessions

- **Session 1:** Stay in sh, use `opencode` (TUI)
- **Session 2:** Run `bash`, do normal work

### 4. Change Default Shell to sh

If you prefer bash for other work, ensure OpenCode runs in a fresh login context. Use a second terminal/SSH session for OpenCode.

## Official Troubleshooting References

- [OpenCode Troubleshooting](https://opencode.ai/docs/troubleshooting/)
- [GitHub Issue #5485](https://github.com/anomalyco/opencode/issues/5485) — TUI stuck on startup

### Quick Fixes from Official Docs

1. **Run with logs:**
   ```bash
   opencode --print-logs --log-level DEBUG
   ```

2. **Clear cache:**
   ```bash
   rm -rf ~/.cache/opencode
   ```

3. **Disable plugins** (in `~/.config/opencode/opencode.json`):
   ```json
   { "plugin": [] }
   ```

4. **Disable autoupdate:**
   ```json
   { "autoupdate": false }
   ```

5. **Check logs:** `~/.local/share/opencode/log/`

6. **Full /tmp partition:** Free space if `/tmp` is full (can cause OpenTUI library load failure)

### Downgrade (if regressions persist)

v1.2.27 has known TUI regressions. Try v1.2.25:

```bash
curl -fsSL https://opencode.ai/install | bash -s -- --version 1.2.25
```

## Environment

- **Default shell:** `/bin/sh` (dash)
- **OpenCode version:** 1.2.27
- **OS:** Ubuntu 24.04.3 LTS
- **Access:** SSH from Windows PowerShell
