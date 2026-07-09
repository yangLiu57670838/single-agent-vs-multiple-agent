# PRD: Single-Agent vs Multi-Agent Comparison on HotpotQA

**Project:** `single-agent-vs-multiple-agent`  
**Version:** 1.0  
**Status:** Draft  
**Last updated:** 2026-07-09

---

## 1. Executive Summary

This project benchmarks two LLM-based question-answering architectures on [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa), a multi-hop QA dataset from Hugging Face. We compare a **single-agent** pipeline (one model handles retrieval, reasoning, and answering) against a **multi-agent** pipeline (specialized agents coordinated by an orchestrator).

The primary research question:

> For multi-hop QA, does splitting work across multiple agents improve answer quality — and if so, at what cost in latency and tokens?

We measure three dimensions on every run:

| Dimension | Question |
|---|---|
| **Quality** | Which architecture produces more correct answers? |
| **Latency** | Which architecture is slower end-to-end? |
| **Token cost** | Which architecture consumes more tokens (and therefore costs more)? |

---

## 2. Background & Motivation

HotpotQA requires finding and reasoning over **multiple supporting documents** to answer a question. Examples include bridge questions ("What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?") and comparison questions ("Were Scott Derrickson and Ed Wood of the same nationality?").

Multi-hop QA is a natural fit for agentic systems:

- A **single agent** can use a ReAct-style loop: read context → reason → answer.
- A **multi-agent** system can delegate sub-tasks (e.g. document selection, fact extraction, cross-document reasoning, answer verification) to specialized agents.

Agent frameworks (Cursor, LangGraph, AutoGen, etc.) increasingly promote multi-agent patterns, but the trade-offs are rarely quantified on a standard benchmark. This repo provides a reproducible, metric-driven comparison.

---

## 3. Goals

### 3.1 Primary Goals

1. **Implement** a single-agent and a multi-agent HotpotQA solver using the same base LLM and prompt budget constraints.
2. **Evaluate** both architectures on a fixed HotpotQA subset with identical preprocessing.
3. **Report** answer quality, latency, and token usage side-by-side with statistical summaries.
4. **Publish** reproducible experiment configs, logs, and result artifacts.

### 3.2 Secondary Goals

- Break down results by question `type` (`bridge` vs `comparison`) and `level` (`easy`, `medium`, `hard`).
- Track per-stage latency in the multi-agent pipeline (retrieval vs reasoning vs synthesis).
- Support multiple LLM backends (e.g. GPT-4o, Claude, open-source models) with per-model cost tables.
- Export results in a format suitable for charts / notebooks.

### 3.3 Non-Goals (v1)

- Training or fine-tuning models on HotpotQA.
- Fullwiki setting with live Wikipedia retrieval (v1 uses **distractor** setting only; see §5.2).
- Supporting-facts prediction as a primary metric (optional stretch goal).
- Production deployment, UI, or real-time serving.
- Human evaluation beyond automated metrics.

---

## 4. Users & Stakeholders

| Stakeholder | Need |
|---|---|
| **ML / AI engineer** | Runnable benchmark scripts, clear configs, comparable logs |
| **Researcher** | Reproducible numbers, ablation-friendly design |
| **Product / platform team** | Evidence for when multi-agent is worth the overhead |
| **Future contributors** | PRD + README + typed experiment outputs |

---

## 5. Dataset

### 5.1 Source

```python
from datasets import load_dataset

ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
# splits: train (90,447), validation (7,405)
```

Hugging Face: [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa)

### 5.2 Configuration: Distractor (v1)

Use the **distractor** config for v1 because:

- Each example ships with **10 paragraphs** (2 gold + 8 distractors) — no external retrieval infra needed.
- Context is self-contained, making agent comparison fair and cheap to run.
- Official validation labels are available for scoring.

**Fullwiki** (retrieve from entire Wikipedia) is deferred to v2.

### 5.3 Schema (per example)

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique example ID |
| `question` | string | Multi-hop question |
| `answer` | string | Gold short answer |
| `type` | string | `bridge` or `comparison` |
| `level` | string | `easy`, `medium`, or `hard` |
| `context` | list | 10 `{title, sentences[]}` paragraphs |
| `supporting_facts` | list | Gold supporting `(title, sent_id)` pairs |

### 5.4 Evaluation Subset

To control cost during development:

| Phase | Split | Size | Purpose |
|---|---|---|---|
| **Dev / smoke** | `validation` | 50 | Fast iteration, CI |
| **Pilot** | `validation` | 500 | Initial comparison |
| **Full run** | `validation` | 7,405 | Final benchmark |

Subset selection must be **deterministic** (fixed seed, saved `subset_ids.json`) so both architectures run on identical questions.

### 5.5 Context Formatting

Convert each example's `context` into a single prompt block:

```
[0] Title A
Sentence 1. Sentence 2. ...

[1] Title B
...
```

Paragraph indices are stable within an example and referenced by agents during reasoning.

---

## 6. Architectures

Both architectures MUST use the **same base model**, **same temperature**, and **same max output tokens** unless running an explicit ablation.

### 6.1 Single-Agent Baseline

One LLM agent receives the question + all 10 context paragraphs and produces a final answer.

**Flow:**

```
Question + Context → [Single Agent: read → reason → answer] → Final Answer
```

**Design choices:**

- ReAct or chain-of-thought prompting encouraged.
- Agent may cite paragraph indices in reasoning (not scored in v1).
- Hard cap on LLM round-trips: **1–3 turns** (configurable).
- No sub-agent delegation.

### 6.2 Multi-Agent Pipeline

Multiple specialized agents coordinated by an **Orchestrator**.

**Proposed v1 topology:**

```
                    ┌─────────────────┐
                    │   Orchestrator   │
                    └────────┬────────┘
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
  │  Retriever   │   │   Reasoner   │   │  Verifier    │
  │  Agent       │   │   Agent      │   │  Agent       │
  └─────────────┘   └──────────────┘   └──────────────┘
```

| Agent | Responsibility |
|---|---|
| **Retriever** | Select top-k relevant paragraphs from the 10 provided |
| **Reasoner** | Perform multi-hop reasoning over selected paragraphs |
| **Verifier** | Check answer consistency; request re-reasoning if needed |
| **Orchestrator** | Route messages, enforce turn limits, emit final answer |

**Constraints (fairness):**

- Multi-agent system gets the **same raw context** as single-agent (no extra documents).
- Total LLM calls capped at a configurable `max_rounds` (default: 6).
- Orchestrator overhead counted in latency and token metrics.

### 6.3 Shared Infrastructure

| Component | Requirement |
|---|---|
| LLM client | Unified wrapper with token counting hooks |
| Prompt templates | Versioned in `prompts/` |
| Config | YAML/JSON per experiment run |
| Logging | Structured JSONL per question |
| Scoring | Shared evaluation module |

---

## 7. Metrics

### 7.1 Quality Metrics

Use standard HotpotQA **answer** metrics:

| Metric | Description |
|---|---|
| **Exact Match (EM)** | Normalized prediction equals gold answer |
| **Token F1** | Token-level overlap F1 after normalization |

**Normalization** (match official HotpotQA eval):

- Lowercase
- Remove punctuation
- Remove articles (`a`, `an`, `the`)
- Strip whitespace

Report aggregate and breakdowns:

- Overall EM / F1
- By `type`: bridge, comparison
- By `level`: easy, medium, hard

**Optional (v1.1):** Supporting Facts EM/F1 and Joint EM/F1.

### 7.2 Latency Metrics

| Metric | Description |
|---|---|
| **End-to-end latency** | Wall-clock time from prompt assembly to final answer (ms) |
| **LLM call latency** | Per API call round-trip time |
| **Orchestration overhead** | Time outside LLM calls (multi-agent only) |
| **p50 / p95 / p99** | Percentiles across the eval set |

Record timestamps at:

- Experiment start / end
- Each LLM request / response
- Each agent handoff (multi-agent)

### 7.3 Token & Cost Metrics

| Metric | Description |
|---|---|
| **Input tokens** | Prompt tokens sent to the LLM |
| **Output tokens** | Completion tokens returned |
| **Total tokens** | Input + output per question |
| **LLM calls** | Number of API round-trips per question |
| **Estimated cost (USD)** | `input_tokens × input_price + output_tokens × output_price` |

Per-agent token breakdown required for multi-agent runs.

**Cost table** (configurable in `configs/pricing.yaml`):

```yaml
gpt-4o:
  input_per_1m: 2.50
  output_per_1m: 10.00
claude-sonnet-4:
  input_per_1m: 3.00
  output_per_1m: 15.00
```

### 7.4 Efficiency Ratios (Derived)

| Ratio | Formula | Insight |
|---|---|---|
| **EM per 1k tokens** | `EM / (total_tokens / 1000)` | Quality efficiency |
| **EM per second** | `EM / latency_seconds` | Speed-quality tradeoff |
| **Cost per correct answer** | `total_cost / correct_count` | Economic efficiency |

---

## 8. Experiment Design

### 8.1 Controlled Variables

| Variable | Control |
|---|---|
| Dataset subset | Identical `subset_ids.json` for both architectures |
| Base model | Same model + version |
| Temperature | Fixed (recommend `0` for reproducibility) |
| Context formatting | Shared formatter |
| Scoring | Shared evaluator |
| Random seed | Fixed and logged |

### 8.2 Independent Variable

- `architecture`: `single` | `multi`

### 8.3 Runs

Each experiment run produces a directory:

```
results/
  {run_id}/
    config.yaml
    subset_ids.json
    single/
      predictions.jsonl
      metrics.json
      traces.jsonl
    multi/
      predictions.jsonl
      metrics.json
      traces.jsonl
    comparison.json        # side-by-side summary
    comparison.md          # human-readable report
```

### 8.4 Prediction Format

Each line in `predictions.jsonl`:

```json
{
  "id": "5a8b57f25542995d3e8f5c3a",
  "question": "...",
  "gold_answer": "yes",
  "predicted_answer": "Yes",
  "em": 1,
  "f1": 1.0,
  "type": "comparison",
  "level": "hard",
  "latency_ms": 3421,
  "input_tokens": 2840,
  "output_tokens": 156,
  "llm_calls": 3,
  "estimated_cost_usd": 0.0086,
  "architecture": "multi",
  "agent_breakdown": {
    "retriever": { "input_tokens": 900, "output_tokens": 40, "latency_ms": 800 },
    "reasoner":  { "input_tokens": 1200, "output_tokens": 80, "latency_ms": 1100 },
    "verifier":  { "input_tokens": 740, "output_tokens": 36, "latency_ms": 900 }
  }
}
```

### 8.5 Comparison Report

`comparison.md` must answer:

1. Which architecture has higher EM / F1?
2. Which is slower (mean, p95)?
3. Which uses more tokens and costs more?
4. Is the quality gap worth the extra cost/latency?
5. Where does multi-agent win or lose (by type/level)?

---

## 9. System Architecture

```
single-agent-vs-multiple-agent/
├── PRD.md
├── README.md
├── configs/
│   ├── experiment.yaml        # model, subset size, architecture toggles
│   └── pricing.yaml           # per-model token pricing
├── data/
│   └── subsets/
│       └── validation_500.json
├── src/
│   ├── dataset/
│   │   └── hotpot.py          # load & format HotpotQA
│   ├── agents/
│   │   ├── single.py          # single-agent solver
│   │   ├── multi/
│   │   │   ├── orchestrator.py
│   │   │   ├── retriever.py
│   │   │   ├── reasoner.py
│   │   │   └── verifier.py
│   │   └── base.py            # shared agent interface
│   ├── llm/
│   │   └── client.py          # unified LLM client + token tracking
│   ├── eval/
│   │   ├── metrics.py         # EM, F1 normalization
│   │   └── compare.py         # generate comparison report
│   └── runner/
│       └── experiment.py      # CLI entry point
├── prompts/
│   ├── single_agent.txt
│   ├── retriever.txt
│   ├── reasoner.txt
│   └── verifier.txt
├── results/                   # gitignored experiment outputs
└── notebooks/
    └── analysis.ipynb         # optional visualization
```

### 9.1 CLI (target)

```bash
# Run single-agent on 500-question pilot
python -m src.runner.experiment \
  --architecture single \
  --subset data/subsets/validation_500.json \
  --model gpt-4o \
  --output results/run_001/single

# Run multi-agent on same subset
python -m src.runner.experiment \
  --architecture multi \
  --subset data/subsets/validation_500.json \
  --model gpt-4o \
  --output results/run_001/multi

# Generate comparison report
python -m src.eval.compare \
  --single results/run_001/single \
  --multi results/run_001/multi \
  --output results/run_001/comparison.md
```

---

## 10. Success Criteria

### 10.1 MVP (v1.0)

- [ ] Load HotpotQA distractor validation split from Hugging Face
- [ ] Single-agent and multi-agent solvers runnable on a 50-question smoke set
- [ ] EM and F1 computed with official normalization
- [ ] Per-question latency and token counts logged
- [ ] Side-by-side `comparison.json` and `comparison.md` generated
- [ ] README with setup and reproduction steps

### 10.2 Benchmark Complete (v1.1)

- [ ] Full 7,405 validation run for both architectures
- [ ] Breakdown by type and level
- [ ] Per-agent token/latency breakdown for multi-agent
- [ ] At least 2 models tested (e.g. GPT-4o + one cheaper model)
- [ ] Documented conclusion: quality vs speed vs cost tradeoff

### 10.3 Hypotheses to Test

| # | Hypothesis |
|---|---|
| H1 | Multi-agent achieves higher EM/F1 on `hard` bridge questions |
| H2 | Multi-agent has ≥2× latency vs single-agent |
| H3 | Multi-agent consumes ≥1.5× tokens due to inter-agent messages |
| H4 | Single-agent is more cost-efficient per correct answer on `easy` questions |
| H5 | Multi-agent advantage diminishes when the base model is stronger |

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| API cost on full validation set | High spend | Staged subsets; cache LLM responses |
| Non-deterministic LLM outputs | Irreproducible results | `temperature=0`; log raw completions; optional rerun |
| Multi-agent prompt bloat | Unfair token comparison | Cap context per agent; log prompt sizes |
| Rate limits | Failed runs | Exponential backoff; resume from checkpoint |
| Answer extraction errors | Scoring noise | Strict output format (`FINAL ANSWER: ...`); parser tests |
| Overfitting prompts to subset | Misleading conclusions | Hold out a separate 500-ID subset for final eval |

---

## 12. Timeline (Suggested)

| Phase | Duration | Deliverable |
|---|---|---|
| **Phase 0 — Setup** | 2–3 days | Dataset loader, LLM client, eval metrics |
| **Phase 1 — Single agent** | 2–3 days | Working single-agent + smoke test results |
| **Phase 2 — Multi agent** | 4–5 days | Orchestrator + 3 sub-agents |
| **Phase 3 — Pilot** | 2 days | 500-question comparison report |
| **Phase 4 — Full benchmark** | 2–3 days | Full validation run + analysis notebook |
| **Phase 5 — Write-up** | 1–2 days | Final report with recommendations |

**Total estimate:** ~2–3 weeks

---

## 13. Open Questions

1. **Which base model(s)?** Start with one capable model (GPT-4o or Claude Sonnet) + one budget model for cost sensitivity analysis.
2. **Retriever strategy:** LLM-based paragraph ranking vs embedding similarity — affects fairness if only multi-agent gets embeddings.
3. **Verifier loop:** Allow one retry on verification failure, or single-pass only?
4. **Answer format:** Free-form vs forced `FINAL ANSWER: <short phrase>` — recommend forced format for reliable EM.
5. **Parallelism:** Run questions concurrently (faster wall-clock) vs sequentially (cleaner latency per question)?

---

## 14. Appendix

### A. HotpotQA Official Metrics Reference

- [HotpotQA Leaderboard](https://hotpotqa.github.io/)
- Answer EM/F1: normalized string match
- Joint EM/F1 (optional): answer correct AND supporting facts correct

### B. Example Multi-Hop Question

> **Q:** What government position was held by the woman who portrayed Corliss Archer in the film *Kiss and Tell*?  
> **A:** Chief of Protocol  
> **Hop 1:** Identify actress who played Corliss Archer → Shirley Temple  
> **Hop 2:** Find government position held by Shirley Temple → Chief of Protocol

### C. Glossary

| Term | Definition |
|---|---|
| **Multi-hop QA** | QA requiring chaining facts from ≥2 documents |
| **Distractor setting** | 10 paragraphs provided; 2 are gold, 8 are noise |
| **EM** | Exact Match — binary correctness |
| **F1** | Token-level F1 between prediction and gold |
| **Orchestrator** | Agent that routes tasks and aggregates outputs |
| **Round-trip** | One LLM request + response pair |
