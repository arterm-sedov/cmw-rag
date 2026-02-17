# LLM Architecture Patterns - Skills & AGENTS.md Update

## Summary

Create 3 focused skills + minimal references section in AGENTS.md, based on validated industry patterns (12-Factor Agents, LangChain Context Engineering, arXiv research).

---

## Changes

### 1. Create `.claude/skills/context-engineering/SKILL.md`

**Subject:** Context window management for multi-stage LLM pipelines

**Content:**
- Reference: 12-Factor Agents Factor 3, LangChain Context Engineering
- Four strategies: Write, Select, Compress, Isolate
- Context types: Instructions, Knowledge, Tools, History
- Context problems: poisoning, distraction, confusion, clash
- Example: Building messages from history for different pipeline stages

---

### 2. Create `.claude/skills/stateless-agent/SKILL.md`

**Subject:** Stateless reducer pattern for agent design

**Content:**
- Reference: 12-Factor Agents Factor 12, Redux pattern
- Core principle: `(State, Observation) => NextStep`
- Benefits: Pause/resume, horizontal scaling, determinism
- Anti-pattern: Agent managing hidden internal state
- Example: Stateless agent function signature

---

### 3. Create `.claude/skills/llm-testing/SKILL.md`

**Subject:** Testing multi-stage LLM pipelines

**Content:**
- Reference: arXiv:2508.20737, Deepchecks best practices
- Three-layer architecture: System Shell → Prompt Orchestration → LLM Inference Core
- Testing by layer: Traditional (shell), Semantic (orchestration), Paradigm shift (core)
- Stage validation: Test context input/output contracts, state transitions
- Mock at boundaries, not internals

---

### 4. Update `AGENTS.md` (append at end)

```markdown
## 📚 Further Reading

External methodologies for production LLM applications:
- [12-Factor Agents](https://humanlayer.dev/12-factor-agents) - Architecture principles
- [Context Engineering](https://blog.langchain.com/context-engineering-for-agents/) - Context management strategies
- [LLM Testing (arXiv:2508.20737)](https://arxiv.org/abs/2508.20737) - Testing multi-stage pipelines

See `.claude/skills/` for detailed guidance on applying these patterns.
```

---

## Files Changed

| File | Action |
|------|--------|
| `.claude/skills/context-engineering/SKILL.md` | Create |
| `.claude/skills/stateless-agent/SKILL.md` | Create |
| `.claude/skills/llm-testing/SKILL.md` | Create |
| `AGENTS.md` | Append references section |
