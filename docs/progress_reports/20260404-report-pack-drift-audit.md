# Report-Pack Drift Audit

## Goal of This Audit

This audit reviews the last few hours of report-pack churn after the earlier smart-sync work. The practical question is not whether the branch moved a lot, but whether later parallel-agent merges:

- reintroduced problems we had already fixed,
- compressed or overwrote nuanced wording that we intentionally kept,
- or added useful new material that still aligns with the original goals of the task.

## Comparison Frame

- Baseline chosen for sanity review: `7b35683`
- Current tip reviewed: `dc3d554`
- Main merge chain after baseline:
  - `137bb0e`
  - `95fde4c`
  - `f75226e`
  - `dc3d554`

These later merges mostly touched the same cluster of files:

- `20260325-research-appendix-a-index-ru.md`
- `20260325-research-appendix-b-ip-code-alienation-ru.md`
- `20260325-research-appendix-d-security-observability-ru.md`
- `20260325-research-appendix-e-market-technical-signals-ru.md`
- `20260325-research-report-methodology-main-ru.md`
- `20260325-research-report-sizing-economics-main-ru.md`
- `20260331-research-executive-unified-ru.md`

## Working Assumptions from Prior Conversation

The earlier merge/sync work established a few implicit editorial goals:

1. Preserve the stronger GitVerse-derived nuance where it was materially richer.
2. Keep the structurally newer and more readable wording where it did not reduce substance.
3. Preserve previously fixed anchor integrity and internal navigation.
4. Avoid letting older agent branches silently roll back the already reviewed smart-sync decisions.
5. Keep genuinely useful new additions if they strengthen the report-pack for C-level, sales, economics, compliance, and sovereign-RF positioning.

## Sanity Verdict

### A. Real regressions were introduced

These are not just stylistic diffs.

- `Methodology` now contains malformed markdown links. This is a direct content-integrity regression.
- `Sizing` again points `Актуальные тренды AI/ML` to a non-existent anchor in `Appendix E`.
- `Appendix A` now references sizing anchors that do not exist in the current sizing file.
- `Sizing` lost source/SKU/date precision in at least one provider-pricing area and replaced it with weaker wording that is harder to defend commercially.
- `Appendix D` lost some of the explicit EU AI Act navigation and sub-structure that previously made the legal framing easier to audit and reuse.

### B. I do not see evidence of total pseudo-conflict collapse

This does **not** look like a catastrophic accidental overwrite of the whole report-pack.

- There are no leftover merge markers in the report-pack.
- Large diffs in `Appendix D` and `Appendix E` are partly due to section reshaping and refactoring, not only content loss.
- Several additions are coherent and appear to come from newer agent work rather than broken merge resolution.

So the branch drift is best described as:

- **localized harmful regressions**, plus
- **useful net-new additions**, plus
- **some compressed legal and pricing nuance**.

## Where the Drift Is Harmful

### 1. Navigation and internal reference integrity

This is the cleanest class of regression because it is objective.

- malformed markdown links in `methodology`
- dead or mismatched anchors between `Appendix A`, `Sizing`, and `Appendix E`

This class should be fixed first because it is low-ambiguity and high-value.

### 2. Commercial and pricing precision

The pricing drift is more subtle than broken links, but more dangerous for executive use.

- a previously more defensible provider/SKU/date framing was flattened
- at least one line now suggests Cloud.ru and SberCloud pricing equivalence in a way that is too coarse for a negotiation-ready document
- this weakens the “sourceable, offer-safe, date-bounded” standard we were trying to maintain

### 3. Legal/compliance navigability

`Appendix D` still contains a lot of legal content, so this is not a blanking-out event. The problem is that some of the detailed structure was removed:

- separate EU AI Act comparative context
- clearer segmentation of timeline/scope/penalties
- sharper navigation around draft-law and gov/KII implications

This reduces reuse value for decision support, even if the remaining text is not plainly wrong.

## Where the Drift Looks Good

These additions appear broadly aligned with the original task goals and should not be thrown away just because they arrived through noisy branches.

### 1. Better external grounding in `methodology`

The added Gartner and BCG references strengthen the argument that:

- AI-ready data is a real gating factor
- people/process change matters more than just model choice
- adoption and realized business value remain meaningfully separated

That direction is consistent with our earlier goal of making the pack stronger for executive decision-making.

### 2. Stronger market and product coverage in `Appendix E`

The additions around:

- `MWS Octapi`
- `Yandex Agent Atelier`
- `Yandex SpeechKit / AI Speech`
- `SaluteSpeech`
- broader voice-layer comparison

look useful and aligned with the market-radar purpose of `Appendix E`, especially for RF enterprise positioning and corporate-agent architecture.

### 3. Better evidence in the executive summary

The added McKinsey/BCG/Yakov-style framing in the unified executive summary generally improves the “why now / what is the business gap” narrative.

That is aligned with the task’s C-level objective, as long as the sources remain clean and the surrounding references stay correct.

## Most Likely Explanation of the Drift

The last few hours look less like “one bad merge” and more like repeated attempts by older side branches to realign against a base that had already moved.

That explains why:

- the same sections were rewritten repeatedly,
- some older wording and older anchors resurfaced,
- some useful new additions landed at the same time,
- and previously reviewed smart-sync decisions were partially overwritten.

In other words, the pattern is consistent with **parallel-agent convergence noise**, not with one intentional editorial decision to revert the pack.

## Recommended Recovery Strategy

Do **not** roll back everything from the last few hours.

Instead, do a selective recovery pass in this order:

1. Restore objective integrity:
   - fix malformed markdown links
   - fix dead anchors and cross-links

2. Restore strong commercial precision:
   - re-check `Sizing` provider/SKU/date wording
   - remove or re-qualify any over-broad equivalence claims

3. Restore lost legal navigation where it was clearly more useful:
   - especially the EU AI Act comparative context and linked structure

4. Preserve aligned net-new additions:
   - `Appendix E` enterprise platform additions
   - voice-layer additions if source quality is acceptable
   - the stronger Gartner/BCG/McKinsey framing in `Methodology` and `Executive Unified`

## Bottom Line

My current judgment is:

- **Yes**, some harmful changes were introduced by conflicting parallel edits.
- **No**, the last few hours are not “mostly garbage”; there are meaningful additions worth keeping.
- The correct move is a **surgical reconcile pass**, not a broad revert.

See the companion file `20260404-report-pack-diff-extracts.md` for focused diff evidence.
