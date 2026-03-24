# Channel Extractions - Storage and Management

## Location

Extraction files are stored outside the repository in:
```
D:/Documents/cmw-rag-channel-extractions/
```

## Files in Extraction Directory

- `.playwright-cli/` - Playwright snapshots for Telegram scraping
- `channel_snapshot.yml` - Telegram channel data
- `ai_machinelearning_channel.md` - Extracted content from @ai_machinelearning_big_data

## Git Ignore Rules

The following are excluded from the repository in `.gitignore`:
```
# Telegram channel extractions (stored in D:/Documents)
.playwright-cli/
channel_snapshot.yml
```

## Why Store Outside Repository

- Extraction files are large and contain raw scraped data
- They can be regenerated from source channels if needed
- Keeping them in `D:/Documents/` prevents repo bloat
- Repository contains only the processed, curated executive summaries in `docs/research/`

## What to Commit

- ✅ Executive summaries in `docs/research/` (processed content)
- ✅ Documentation in `docs/research/extractions.md` (this file)
- ❌ Raw extraction files (stored in D:/Documents)