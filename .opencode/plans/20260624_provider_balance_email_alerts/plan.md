# Provider Balance Email Alerts

## Summary

Add a **standalone scheduled job** in `cmw-rag` that monitors **shared `.env` provider keys** against explicit thresholds and emails operators on first `low` / `depleted` transition, plus one recovery email on return to `ok`.

Operator emails may include numeric balances. **Never** expose those numbers in Gradio UI, APIs, streamed responses, or default logs. No billing HTTP from request handlers.

**Sibling feature (different repo):** cmw-platform-agent — `docs/plans/20260624_provider_account_stats_plan.md` (Statistics-tab provider account stats). UI coarse health for `.env` keys; detailed numbers only for Config-tab session keys. **Do not import** platform-agent code; mirror adapter semantics and threshold env names only.

## Repo facts (verified 2026-06-24)

| Topic | Decision |
|---|---|
| OpenRouter settings | `openrouter_api_key`, `openrouter_base_url` — already in `Settings` |
| Polza / GigaChat settings | **Not in cmw-rag today** — add optional fields (see Configuration) |
| HTTP client | `requests>=2.28.0` in `rag_engine/requirements.txt` — use `requests`, bounded timeout |
| Script pattern | `rag_engine/scripts/check_provider_balances.py` + `argparse` (like `sync_mkdocs_corpus.py`) |
| Script invocation | `%h/cmw-rag/.venv/bin/python rag_engine/scripts/check_provider_balances.py` — **not** `python -m` (no `scripts/__init__.py`) |
| systemd | `systemd/` — mirror `cmw-rag-corpus-sync.service` / `.timer` (`%h/cmw-rag`, journal output) |
| Settings import | `settings = Settings()` at module load — **new alert fields must be optional with defaults**; validate only when `key_balance_alerts_enabled=true` inside the checker script |

Pre-flight: read `AGENTS.md`, run `git status --short`, avoid unrelated changes.

## Scope

**In:** OpenRouter, Polza, GigaChat adapters; alert + SMTP config; `rag_engine/ops/provider_balance_alerts.py`; checker script; tests; `systemd/cmw-rag-provider-balance-alerts.{service,timer}`; `.env-example` + README.

**Out:** Session/Config-tab keys; Gradio/UI billing calls; OpenAI org usage; importing `cmw-platform-agent`; new email SDKs (stdlib `smtplib` only).

## Pinned policies

| Policy | Value |
|---|---|
| Threshold inference | Never — missing threshold ⇒ provider skipped (`unavailable`), no low/depleted alert |
| Classification | `remaining <= 0` ⇒ `depleted`; `0 < remaining <= threshold` ⇒ `low` (inclusive); else `ok` |
| `depleted → low` | **Send `low` alert** (partial recovery, still below threshold) |
| Recovery emails | **Always on** in v1 (`RECOVERY_ENABLED = True` module constant) |
| Provider timeout | **8.0 s** module constant (`PROVIDER_TIMEOUT_SECONDS`) |
| Duplicate alerts | Once per provider per notified state; `last_notified_status` unchanged on `error`/`unavailable`/SMTP failure |
| Recovery state write | Set `last_notified_status = "ok"` + `last_recovery_notified_at` |
| Enabled, zero providers | Exit **1** before HTTP if no provider has **both** key and threshold |
| Provider `error`/`unavailable` | Exit **0** after handled run (do not page ops for transport blips in v1) |
| SMTP / state failure | Exit **2** |
| Invalid alert config when enabled | Exit **1** |

## Files

```text
.env-example                                          # alert + provider threshold block
rag_engine/config/settings.py                         # optional alert + polza/gigachat fields
rag_engine/ops/__init__.py
rag_engine/ops/provider_balance_alerts.py             # models, adapters, state, email, SMTP
rag_engine/scripts/check_provider_balances.py         # CLI entry
rag_engine/tests/test_provider_balance_alerts_config.py
rag_engine/tests/test_provider_balance_alerts_providers.py
rag_engine/tests/test_provider_balance_alerts_state.py
rag_engine/tests/test_provider_balance_alerts_email.py
rag_engine/tests/test_check_provider_balances_cli.py
rag_engine/tests/fixtures/provider_balance/           # JSON mocks per provider
systemd/cmw-rag-provider-balance-alerts.service
systemd/cmw-rag-provider-balance-alerts.timer
README.md                                             # ops section when behavior/commands change
```

## Configuration

### Alert + SMTP (`Settings` — all optional unless noted)

| Python field | Env var | Default | Validated when enabled |
|---|---|---|---|
| `key_balance_alerts_enabled` | `KEY_BALANCE_ALERTS_ENABLED` | `false` | — |
| `key_balance_alert_recipients` | `KEY_BALANCE_ALERT_RECIPIENTS` | `""` | non-empty, parseable list |
| `key_balance_alert_from` | `KEY_BALANCE_ALERT_FROM` | `""` | non-empty |
| `key_balance_alert_state_path` | `KEY_BALANCE_ALERT_STATE_PATH` | `var/provider_balance_alert_state.json` | non-empty path |
| `smtp_host` | `SMTP_HOST` | `""` | non-empty |
| `smtp_port` | `SMTP_PORT` | `587` | — |
| `smtp_username` | `SMTP_USERNAME` | `""` | optional |
| `smtp_password` | `SMTP_PASSWORD` | `""` | optional |
| `smtp_use_tls` | `SMTP_USE_TLS` | `true` | — |

Recipient parsing: comma-separated, trim, drop empty, de-dupe case-insensitively; reject control chars / missing `@`.

### Provider keys + thresholds (add to `Settings`)

| Python field | Env var | Default | Notes |
|---|---|---|---|
| *(existing)* `openrouter_api_key` | `OPENROUTER_API_KEY` | required today | monitor when threshold set |
| *(existing)* `openrouter_base_url` | `OPENROUTER_BASE_URL` | required today | strip trailing `/`; credits at `{origin}/api/v1/credits` |
| `polza_api_key` | `POLZA_API_KEY` | `None` | align name with platform-agent |
| `polza_base_url` | `POLZA_BASE_URL` | `https://polza.ai/api/v1` | |
| `gigachat_api_key` | `GIGACHAT_API_KEY` | `None` | authorization key (Base64 credentials) |
| `gigachat_scope` | `GIGACHAT_SCOPE` | `GIGACHAT_API_PERS` | |
| `gigachat_verify_ssl` | `GIGACHAT_VERIFY_SSL` | `false` | match platform-agent default |
| `gigachat_base_url` | `GIGACHAT_BASE_URL` | `https://gigachat.devices.sberbank.ru/api/v1` | balance under `{base}/balance` |
| `openrouter_low_balance_threshold_usd` | `OPENROUTER_LOW_BALANCE_THRESHOLD_USD` | `None` | `Decimal`; unset ⇒ skip OpenRouter alerting |
| `polza_low_balance_threshold_rub` | `POLZA_LOW_BALANCE_THRESHOLD_RUB` | `None` | `Decimal` |
| `gigachat_low_token_threshold` | `GIGACHAT_LOW_TOKEN_THRESHOLD` | `None` | `int` |

Place the `.env-example` block after LLM provider keys. Document that Polza/GigaChat keys are optional and only needed when monitoring those providers.

## Domain model

```python
ProviderName = Literal["openrouter", "polza", "gigachat"]
BalanceStatus = Literal["ok", "low", "depleted", "unavailable", "error"]
AlertKind = Literal["low", "depleted", "recovered"]

@dataclass(frozen=True)
class ProviderBalanceResult:
    provider: str
    status: BalanceStatus
    checked_at: datetime
    balance: Decimal | int | None = None
    currency: str | None = None
    used: Decimal | int | None = None
    limit: Decimal | int | None = None
    remaining: Decimal | int | None = None
    reset_period: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    error_class: str | None = None
    error_message: str | None = None  # sanitized; no secrets/URLs with tokens
```

Pure helpers (test without I/O):

```python
def classify_decimal_balance(value: Decimal | None, threshold: Decimal | None) -> BalanceStatus: ...
def classify_token_balance(value: int | None, threshold: int | None) -> BalanceStatus: ...
def decide_alert_event(
    last_notified_status: str | None,
    current: ProviderBalanceResult,
    *,
    recovery_enabled: bool = True,
) -> AlertKind | None: ...
```

**Provider selection:** include provider iff `key` non-empty **and** threshold set; else log `unavailable` and skip HTTP (except dry-run may still list as skipped).

## Provider API contracts

Align with cmw-platform-agent `docs/plans/20260624_provider_account_stats_plan.md`. Store fixtures under `rag_engine/tests/fixtures/provider_balance/`.

### OpenRouter

| | |
|---|---|
| Credits | `GET {openrouter_origin}/api/v1/credits` — header `Authorization: Bearer {openrouter_api_key}` |
| Key (optional diagnostics) | `GET {openrouter_origin}/api/v1/key` — same auth; `limit=null` is valid |
| Remaining | `remaining = total_credits - total_usage` when both present (`Decimal`) |
| Currency | `USD` |
| Auth errors | `401`/`403` ⇒ `error`, sanitized message |
| Malformed / missing numbers | `unavailable` or `error` |

Fixture `openrouter_credits_ok.json`:

```json
{"data": {"total_credits": 10.0, "total_usage": 7.87}}
```

### Polza

| | |
|---|---|
| Balance | `GET {polza_base_url}/balance` — `Authorization: Bearer {polza_api_key}` |
| Map | `amount` → `remaining`/`balance` (RUB); `spentAmount` → `used` if present |
| Transport | Timeout/TLS/connectivity ⇒ `error`; **do not abort** other providers (known flaky host) |
| Malformed JSON | `error` |

Fixture `polza_balance_ok.json`:

```json
{"amount": 470.02686607, "spentAmount": 29.97}
```

### GigaChat

| | |
|---|---|
| OAuth | `POST https://ngw.devices.sberbank.ru:9443/api/v2/oauth` — `Authorization: Basic {gigachat_api_key}`, body `scope={gigachat_scope}`, header `RqUID: {uuid4}` |
| Balance | `GET {gigachat_base_url}/balance` — `Authorization: Bearer {access_token}` |
| SSL | Honor `gigachat_verify_ssl` (do not silently disable) |
| Packages | Normalize to `details["packages"]`; classify on **lowest non-null remaining** across packages |
| Zero tokens | Valid ⇒ `depleted` |
| `403` / no packages | `unavailable` |
| OAuth failure | `error`, no token in logs |

Fixture `gigachat_balance_ok.json` (shape per Sber API; adjust in tests to match real response):

```json
{"balance": [{"package": "default", "balance": 50000}]}
```

Implement `fetch_openrouter_balance`, `fetch_polza_balance`, `fetch_gigachat_balance` — each catches exceptions and returns normalized results.

## Alert transitions

| `last_notified_status` | Current | Email | Kind |
|---|---|---:|---|
| `null` | `ok` | no | — |
| `null` | `low` | yes | `low` |
| `null` | `depleted` | yes | `depleted` |
| `low` | `low` | no | — |
| `low` | `depleted` | yes | `depleted` |
| `low` | `ok` | yes | `recovered` |
| `depleted` | `depleted` | no | — |
| `depleted` | `low` | yes | `low` |
| `depleted` | `ok` | yes | `recovered` |
| any | `unavailable` / `error` | no | — |

On SMTP success: update `last_notified_status` + `last_notified_at`. On failure: update `last_observed_*` only. On recovery success: `last_notified_status = "ok"`, set `last_recovery_notified_at`.

## State file

Path: `key_balance_alert_state_path` (default `var/provider_balance_alert_state.json`).

```json
{
  "version": 1,
  "providers": {
    "openrouter": {
      "last_observed_status": "low",
      "last_observed_at": "2026-06-24T12:00:00Z",
      "last_notified_status": "low",
      "last_notified_at": "2026-06-24T12:00:01Z",
      "last_recovery_notified_at": null,
      "last_error_class": null,
      "consecutive_error_count": 0
    }
  }
}
```

- No numeric balances, raw responses, keys, or tokens in state.
- Missing file ⇒ empty state. Malformed ⇒ log, rename to `*.corrupt.<timestamp>`, continue empty.
- Atomic write: temp file in same dir → `fsync` → replace.

## Email

- Transport: `smtplib.SMTP` + `STARTTLS` when `smtp_use_tls`; `login` only if `smtp_username` set.
- Subjects: `[cmw-rag] Provider balance LOW: OpenRouter` (DEPLETED / RECOVERED variants).
- Body: plain text with provider, status, checked_at, threshold, remaining, optional used/limit/details, state path; **no keys/tokens**.
- Logs: provider, status, alert kind, send ok/fail, error class — **not** numeric balances.

## Checker script

`rag_engine/scripts/check_provider_balances.py`:

```text
--provider openrouter|polza|gigachat   # optional filter
--dry-run                              # no SMTP, no state write for notifications
--include-numeric                      # local debug only; never in systemd
```

Flow: load settings → if disabled, log and exit 0 → `validate_alert_config()` → load state → for each selected provider fetch/classify/decide/send → atomic state save → sanitized summary log.

## systemd

`systemd/cmw-rag-provider-balance-alerts.service`:

```ini
[Unit]
Description=cmw-rag provider balance email alerts
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=%h/cmw-rag
ExecStart=%h/cmw-rag/.venv/bin/python rag_engine/scripts/check_provider_balances.py
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cmw-rag-balance-alerts

[Install]
WantedBy=default.target
```

`systemd/cmw-rag-provider-balance-alerts.timer`:

```ini
[Unit]
Description=Hourly cmw-rag provider balance email alerts
Requires=cmw-rag-provider-balance-alerts.service

[Timer]
OnCalendar=hourly
RandomizedDelaySec=300
Persistent=true

[Install]
WantedBy=timers.target
```

Ensure deploy user can write `var/` (or configured state path).

## Implementation (TDD)

### Phase 1 — Settings

1. Add optional alert, SMTP, polza/gigachat, threshold fields to `Settings`.
2. Add `parse_alert_recipients(raw) -> list[str]` and `validate_alert_config(settings) -> None | str` (error message).
3. Update `.env-example`.
4. **First test:** `test_key_balance_alerts_disabled_by_default`.

### Phase 2 — Core ops (no HTTP)

1. `rag_engine/ops/provider_balance_alerts.py` — models, classify helpers, `decide_alert_event`, state load/save, email render.
2. **First test:** `test_classify_decimal_balance_at_threshold_is_low`.
3. **First test:** `test_depleted_to_low_sends_low_alert`.

### Phase 3 — Provider adapters

1. Implement three `fetch_*` functions with `requests` + mocks from fixtures.
2. **First test:** `test_fetch_openrouter_balance_maps_remaining_usd`.
3. One provider failure must not block others.

### Phase 4 — SMTP + script

1. `send_alert_email(...)` with injectable SMTP for tests.
2. Wire `main()` / `parse_args()` / exit codes in checker script.
3. **First test:** `test_disabled_alerts_exit_zero_without_smtp`.

### Phase 5 — systemd + docs

1. Add unit files under `systemd/`.
2. README ops subsection: enable flag, timer install, dry-run, state file location, rollback.

## Verification

```powershell
# from repo root
.venv\Scripts\Activate.ps1
git status --short
python -m pytest rag_engine/tests -k provider_balance_alert -q
ruff check rag_engine/config/settings.py rag_engine/ops/provider_balance_alerts.py rag_engine/scripts/check_provider_balances.py rag_engine/tests/test_provider_balance_alerts_config.py rag_engine/tests/test_provider_balance_alerts_providers.py rag_engine/tests/test_provider_balance_alerts_state.py rag_engine/tests/test_provider_balance_alerts_email.py rag_engine/tests/test_check_provider_balances_cli.py
$env:KEY_BALANCE_ALERTS_ENABLED="false"
python rag_engine/scripts/check_provider_balances.py --dry-run
```

Security fixtures: assert `sk-test-secret`, `Bearer abc123`, `smtp-password`, `oauth-token` never appear in logs, state JSON, or default CLI output.

## Rollback

1. `KEY_BALANCE_ALERTS_ENABLED=false`
2. `systemctl disable --now cmw-rag-provider-balance-alerts.timer`
3. Keep state file unless operators want alerts to re-fire from scratch

## Definition of done

- [ ] Optional settings do not break app startup when alert SMTP is unset
- [ ] Checker runs disabled (exit 0) and enabled dry-run without secrets in logs
- [ ] All three adapters covered by mocked tests; alignment with platform-agent threshold semantics
- [ ] Deduplicated operator emails with numeric detail in body only
- [ ] No Gradio/request-path billing HTTP
- [ ] systemd units match `cmw-rag-corpus-sync` conventions
- [ ] `ruff check` + focused pytest green; README updated
