# Progress Report: Financial Contradictions Discovery (Technology Transfer Research)
Date: 2026-04-09
Task: Find financial contradictions in the technology transfer report pack.

## Executive Summary
I have identified significant internal contradictions in the financial data within the report pack. These contradictions primarily stem from two different calculation methods being used in different parts of the report:
1. **Method A (Clean-USD):** Round figures in USD ($2.5k, $10k, $100k) converted to RUB at a fixed rate (1 USD = 85 RUB). Used in the summary TCO table (4.10.17).
2. **Method B (Market-RUB):** Actual market prices and sizing examples that include total cost of ownership factors, assembly, and local pricing overhead. Used in the detailed sections (4.11 and 4.10.2-4).

## Identified Contradictions

### 1. Local (On-Prem) CapEx - Significant Underselling in Summary
| Segment | Summary Table (4.10.17) | Detailed Sections (4.11.1-3 / 4.10.2-4) | Discrepancy |
| :--- | :--- | :--- | :--- |
| **Small** | 212 500 ₽ | 400 000 – 600 000 ₽ | **-47% to -65%** |
| **Medium** | 850 000 ₽ | 1 200 000 – 2 000 000 ₽ | **-30% to -57%** |
| **Large** | 8 500 000 ₽ | от ~12 000 000 ₽ | **-29%** |

### 2. Cloud OpEx - Monthly vs Annual Mismatch
- **Start-up cost contradiction:** Section 4.6.2 states cloud starts from **212 500 ₽/month**. However, the TCO table (4.10.17) lists the annual OpEx for a "Small Cloud" as 1 020 000 ₽, which averages to **85 000 ₽/month**.
- **Impact:** The TCO summary underestimates entry-level cloud costs by **2.5x** compared to the "start from" claim.

### 3. Cloud Large-Scale Tier Costs
- **TCO Summary (4.10.17):** Lists Large Cloud OpEx as 10 200 000 ₽/year (**850 000 ₽/month**).
- **Cloud.ru (4.10.6):** Minimal recommended inference configuration (4xV100) costs **721 000 ₽/month**. A more standard A100 configuration (5xA100) costs **1 158 000 ₽/month**.
- **Yandex Cloud (4.10.7):** Large instance (8xA100) costs **1 460 000 – 2 555 000 ₽/month**.
- **Impact:** The TCO summary potentially underestimates high-performance cloud costs by **up to 3x**.

### 4. GPU Unit Rates (Hourly)
- **Selectel/Yandex H100 rate:** Table 4.9.2 lists **900 – 2 200 ₽/hour**.
- **Cloud.ru H100 rate calculation:** Table 4.10.6 lists 5xH100 at 2 745 ₽/hour, which is **549 ₽/hour/card**.
- **Difference:** **-39% to -75%** inconsistency across hardware provider tables.

### 5. Extreme Outlier - NVIDIA H200
- **Section 4.6.2:** Mentions NVIDIA H200 cost up to **50 000 000 ₽**.
- **Context:** In Table 4.9.2, H100 is estimated at ~2.1M - 3.4M ₽. While H200 is newer, a 15x-20x price jump for a single card is unverified and contradicts the "Small/Medium/Large" tiering elsewhere.

## Next Steps
1. Perform deep research into the actual market availability and pricing of the H200 and Blackwell (B200) in Russia for 2026.
2. Resolve the discrepancy between the $2,500/$10,000/$100,000 "Method A" and the market reality "Method B".
3. Harmonize the TCO table (4.10.17) with the detailed sizing examples (4.11).
4. Update cloud rates to reflect current provider tariffs consistently across all sections.

---
**Status:** In Progress
**Artifacts:** Detailed list of contradictions identified.
**Recommendations:** Align summary TCO tables with market-based RUB figures from sections 4.10.2-4 and 4.11. Use actual cloud provider monthly billing instead of arbitrary annual splits.
