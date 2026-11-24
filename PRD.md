# PRD: Single-Agent vs Multi-Agent Comparison on HotpotQA

**Project:** `single-agent-vs-multiple-agent`  
**Version:** 1.0  
**Status:** Draft  
**Last updated:** 2026-07-11

---

## 1. Executive Summary

This project benchmarks two LLM-based question-answering architectures on [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa), a multi-hop QA dataset from Hugging Face. We compare a **single-agent** pipeline (one model handles retrieval, reasoning, and answering) against a **multi-agent** pipeline built as a **custom workflow of sub-agents**, coordinated by a **code-based supervisor** (not an LLM orchestrator).

The primary research question:

> For multi-hop QA, does splitting work across multiple agents — each seeing only task-relevant context — improve answer quality and reduce hallucination — and if so, at what cost in latency and tokens?

This repo tests **one** reason to use multiple agents: **Point 1, scoped context** (§2.1). The final report may also name other reasons to use multi-agent (§2.6) and reasons to keep a single agent (§2.7). The numbers only speak to Point 1, and to delay and cost on this small task (§9.5).

See **§2 What We Want to Prove** for the full hypothesis, expected outcomes, and proof criteria.

We measure three dimensions on every run. Quality tests Point 1. **Delay and token cost are the trade-off**: more agents mean more round-trips and more tokens, so the final report must say whether that penalty is worth the quality gain (§2.3, §9.5).

| Dimension | Question |
|---|---|
| **Quality** | Which architecture produces more correct answers? |
| **Delay** | How much slower is multi-agent, and where does the time go? |
| **Token cost** | How much more does multi-agent spend, and what does each correct answer cost? |

---

## 2. What We Want to Prove

### 2.1 Core Hypothesis — Point 1: Scoped Context

This repo tries to prove **one** claim:

> **Point 1.** Irrelevant material that stays in the prompt becomes noise and interferes with the decision. Split the job so each stage sees only the context required for its own task. The coordinator then uses each stage's short result, not the raw inputs that stage read.

Why this shows up in a single agent: the model has no separate memory. Everything it has been shown — every paragraph, every earlier tool result — is still in the next call unless something outside the model removes it. Deciding what can be safely dropped, or summarizing the trace and keeping only the summary, is its own policy and spends extra tokens. A sub-agent can do the wide read and return a filtered result (for example, the ten places auth is checked, not fifty files of source). The lead reasons over that result only.

**What this benchmark actually runs** is a small, fixed version of that idea. It is not a 200-file sweep, not a context window filling up, and not a transcript that grows across dozens of tool calls. HotpotQA distractor gives every question **10 paragraphs** (2 gold, 8 distractors). The single agent is handed all 10 up front, in 1–3 turns. The multi-agent arm narrows context between stages:

| Architecture | Context exposure |
|---|---|
| **Single-agent** | All 10 paragraphs in every LLM call — retrieval, reasoning, and answering share the same noisy context |
| **Multi-agent** | Code supervisor runs a fixed workflow; retriever sees all 10 and returns top-k indices → reasoner sees those paragraphs only → verifier sees the answer + cited sentences only |

The retriever is the sub-agent that "reads everything and returns the hits." The supervisor drops the other paragraphs before the reasoner is called, so those distractors never enter the reasoner's prompt. That filter is the mechanism under test — not conversation compaction inside one agent.

We measure whether that noise reduction improves answer quality, hallucination, and citation, and what extra latency and tokens the split costs (§2.2–§2.4).

### 2.2 Expected Outcomes (if hypothesis holds)

| Claim | Rationale | How we measure it |
|---|---|---|
| **Less hallucination** | Focused context reduces inventing facts not grounded in provided paragraphs | Trace audit: flag answers citing sentences/entities absent from context; hallucination rate = flagged / total |
| **Fewer mis-citations** | Each agent cites only from its scoped context, reducing wrong-paragraph attribution | Supporting Facts EM/F1 vs gold `supporting_facts`; paragraph index accuracy in agent traces |
| **Higher accuracy** | Less noise → more reliable multi-hop reasoning, especially on bridge questions | Answer EM and F1 overall; breakdown by `type` (`bridge`, `comparison`) and `level` (`easy`, `medium`, `hard`) |

### 2.3 Trade-off We Will Evaluate: Delay and Cost

More agents means more LLM round-trips and more tokens. The split has to buy something. We **will evaluate delay and cost on every run**, for both architectures, on the same questions. The final report (`comparison.md`, §9.5) **must explain** those numbers in prose: what the penalty is, where it comes from, and whether the Point 1 gains pay for it.

| Trade-off | What we evaluate | What the final report must explain |
|---|---|---|
| **Delay** | End-to-end latency (mean, p50, p95); per-stage time (retrieve, reason, verify); orchestration overhead (wall-clock outside LLM calls) | How much slower multi-agent is, where the time goes, and the latency ratio (multi-agent p95 / single-agent p95). Expected direction: slower, because of extra LLM rounds (H4, about ≥2×). |
| **Cost** | Input and output tokens per question; LLM call count; estimated USD from `configs/pricing.yaml` | How much more multi-agent spends, which stage spends the tokens, cost per correct answer, and EM per 1k tokens. Expected direction: higher, because each stage sends its own prompt (H5, about ≥1.5× tokens). |

The explanation is required either way. If quality improves, the report says whether the gain is worth the extra delay and cost (§2.4). If it does not, the report says the split still costs more and runs slower, so prefer the single agent for this task (§2.5).

This evaluation is delay and token cost only. Supervisor time is included as orchestration overhead. Design effort, and new failure modes introduced by the split, are not scored.

### 2.4 Proof Criteria

We consider the core hypothesis **supported** if multi-agent shows meaningful improvement on **at least two of**:

1. **Answer EM / F1** — higher overall, or specifically on `hard` / `bridge` subsets
2. **Lower hallucination rate** — fewer unsupported claims in audited traces
3. **Fewer mis-citations** — higher Supporting Facts EM/F1 (enabled in v1.1)

And we publish a clear verdict on **whether the gain is worth it**, using:

- **Cost per correct answer** = total cost / number of EM-correct answers
- **Latency ratio** = multi-agent p95 / single-agent p95
- **Quality-per-1k-tokens** = EM / (total_tokens / 1000)

### 2.5 Null Hypothesis

> Multi-agent context splitting does **not** materially improve answer quality over a well-prompted single agent, but still costs more and runs slower.

If results match the null hypothesis, the conclusion is: **prefer single-agent** for this task unless other factors (debuggability, modularity) matter.

### 2.6 Other Reasons to Use Multi-Agent — Named, Not Tested

Point 1 is not the only reason a multi-agent system can be better. The final report may say so in a few sentences, then state that **this test repo only tries to prove Point 1**. The table below is that scope note. None of these rows are hypotheses, and none of them should be read off the HotpotQA numbers.

| # | Other situation where multi-agent can help | Why this repo does not prove it |
|---|---|---|
| 2 | Independent jobs run in parallel and cut wall-clock time | The workflow is sequential (retrieve → reason → verify). Each step needs the previous result. The expected latency result is slower, not faster (§2.3). |
| 3 | A reviewer with a fresh context catches blind spots the writer missed | The verifier does see a narrower payload, but any gain is confounded with the retriever filter. v1 does not ablate verifier-on vs verifier-off, so it is not a separate claim. |
| 4 | Different jobs can use different tools, permissions, or models | Every sub-agent uses the same base model, temperature, and output cap (§7.2.6). There are no tools and no permission boundaries. |
| 5 | A sub-agent can keep working in the background without blocking other jobs | The supervisor is a synchronous state machine. Latency is end-to-end per question. |

### 2.7 When to Keep a Single Agent

These are reasons to stay with one agent. The final report names all four in a short note (§9.5). Only two of them are things this benchmark's numbers can speak to. They are not extra hypotheses beyond H4–H6 and §2.5.

| # | Keep a single agent when | What this repo does with it |
|---|---|---|
| 1 | **The steps are tightly linked.** Each step depends on the last (debugging, step-by-step refactoring, most coding). Handing work between agents loses the subtle context that matters. | **Name only.** Retrieve → reason → verify is tightly linked, and a short handoff can drop a fact the next stage needed. Debugging and refactoring are not this dataset. A HotpotQA win or loss does not prove this rule. |
| 2 | **The task is small.** If it fits easily in one context, splitting it only adds overhead. | **The numbers speak to this.** Ten paragraphs fit in one prompt. On `easy` questions, the report uses H6 plus the delay and cost verdict (§2.3): did the split only add overhead? If `hard` / `bridge` still improves, fitting in one context was not enough — the 8 distractors were still noise. |
| 3 | **Consistency matters.** Several agents editing the same files or making design choices separately will conflict and drift. | **Name only.** These agents do not edit shared files or make design choices in parallel. The supervisor is code, so routing does not drift. |
| 4 | **Coordination costs more than the work.** Each new agent starts from nothing and has to rebuild context you already have. Briefs lose information, and summaries can hide mistakes. | **The cost half, yes.** Each stage starts a new prompt and re-sends the question plus its slice. That extra delay and those extra tokens are §2.3. A brief that hid a mistake is not its own metric. It shows up only if the answer gets worse. |

The report's "prefer the single agent" sentence stays §2.5: quality did not move enough to pay for the extra delay and cost, especially on easy questions. That is the evidence for rows 2 and 4. Rows 1 and 3 are the wider rule, not results of this run.

---

## 3. Background & Motivation

HotpotQA requires finding and reasoning over **multiple supporting documents** to answer a question. Examples include bridge questions ("What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?") and comparison questions ("Were Scott Derrickson and Ed Wood of the same nationality?").

Multi-hop QA is a natural fit for agentic systems:

- A **single agent** can use a ReAct-style loop: read context → reason → answer.
- A **multi-agent** system uses a **custom workflow of sub-agents** (retriever, reasoner, verifier) coordinated by a **code-based supervisor** that filters context between steps.

Agent frameworks (Cursor, LangGraph, AutoGen, etc.) increasingly promote multi-agent patterns, but the trade-offs are rarely quantified on a standard benchmark. This repo provides a reproducible, metric-driven comparison.

---

## 4. Goals

### 4.1 Primary Goals

1. **Implement** a single-agent and a multi-agent HotpotQA solver using the same base LLM and prompt budget constraints.
2. **Prove or disprove** the context-isolation hypothesis in §2 (accuracy, hallucination, mis-citation) and **evaluate the delay and cost trade-off** on the same runs (§2.3).
3. **Evaluate** both architectures on a fixed HotpotQA subset with identical preprocessing.
4. **Explain in the final report** the quality result and the delay and cost penalty: how much slower, how much more expensive, where the time and tokens go, and whether the split is worth it (§9.5).
5. **Publish** reproducible experiment configs, logs, and result artifacts.

### 4.2 Secondary Goals

- Break down results by question `type` (`bridge` vs `comparison`) and `level` (`easy`, `medium`, `hard`).
- Track per-stage latency in the multi-agent pipeline (retrieval vs reasoning vs synthesis).
- Support multiple LLM backends (e.g. GPT-4o, Claude, open-source models) with per-model cost tables.
- Export results in a format suitable for charts / notebooks.

### 4.3 Non-Goals (v1)

- Training or fine-tuning models on HotpotQA.
- Fullwiki setting with live Wikipedia retrieval (v1 uses **distractor** setting only; see §6.2).
- Supporting-facts prediction as a primary metric (optional stretch goal).
- Production deployment, UI, or real-time serving.
- Human evaluation beyond automated metrics.

---

## 5. Users & Stakeholders

| Stakeholder | Need |
|---|---|
| **ML / AI engineer** | Runnable benchmark scripts, clear configs, comparable logs |
| **Researcher** | Reproducible numbers, ablation-friendly design |
| **Product / platform team** | Evidence for when multi-agent is worth the overhead |
| **Future contributors** | PRD + README + typed experiment outputs |

---

## 6. Dataset

### 6.1 Source

```python
from datasets import load_dataset

ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
# splits: train (90,447), validation (7,405)
```

Hugging Face: [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa)

### 6.2 Configuration: Distractor (v1)

Use the **distractor** config for v1 because:

- Each example ships with **10 paragraphs** (2 gold + 8 distractors) — no external retrieval infra needed.
- Context is self-contained, making agent comparison fair and cheap to run.
- Official validation labels are available for scoring.

**Fullwiki** (retrieve from entire Wikipedia) is deferred to v2.

### 6.3 Schema (per example)

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique example ID |
| `question` | string | Multi-hop question |
| `answer` | string | Gold short answer |
| `type` | string | `bridge` or `comparison` |
| `level` | string | `easy`, `medium`, or `hard` |
| `context` | list | 10 `{title, sentences[]}` paragraphs |
| `supporting_facts` | list | Gold supporting `(title, sent_id)` pairs |

### 6.4 Evaluation Subset

To control cost during development:

| Phase | Split | Size | Purpose |
|---|---|---|---|
| **Dev / smoke** | `validation` | 50 | Fast iteration, CI |
| **Pilot** | `validation` | 500 | Initial comparison |
| **Full run** | `validation` | 7,405 | Final benchmark |

Subset selection must be **deterministic** (fixed seed, saved `subset_ids.json`) so both architectures run on identical questions.

### 6.5 Context Formatting

Convert each example's `context` into a single prompt block:

```
[0] Title A
Sentence 1. Sentence 2. ...

[1] Title B
...
```

Paragraph indices are stable within an example and referenced by agents during reasoning.

---

## 7. Architectures

Both architectures MUST use the **same base model**, **same temperature**, and **same max output tokens** unless running an explicit ablation.

### 7.1 Single-Agent Baseline

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

### 7.2 Multi-Agent Architecture: Custom Workflow + Sub-Agents + Code Supervisor

The multi-agent arm uses a **fixed custom workflow** of **specialized sub-agents**, coordinated by a **code-based supervisor** (`orchestrator.py`). The supervisor is plain Python — it does **not** make LLM calls and does **not** see or reason over paragraph content. It only manages state, filters context, invokes sub-agents in order, and logs metrics.

#### 7.2.1 Pattern Decision

| Pattern | v1 decision | Rationale |
|---|---|---|
| **Custom workflow** | **Adopt** | HotpotQA has a natural fixed pipeline (select → reason → verify); reproducible and easy to benchmark per-stage |
| **Sub-agents** | **Adopt** | Separate LLM calls per stage enable different prompt payloads per task |
| **Code-based supervisor** | **Adopt** | Enforces context isolation policy (§2.1); no extra LLM tokens; deterministic routing |
| **Handoff** | **Partial** | Used only for verifier → reasoner retry (max 1 retry) |
| **Router (LLM)** | **Reject** | Non-deterministic; adds LLM calls that confound quality vs cost comparison |
| **Skills (single agent)** | **Reject** | One shared context window; does not test context-isolation hypothesis |
| **LLM supervisor** | **Reject** | Would see full context and add token overhead; undermines §2.1 |

> **Context isolation is enforced by supervisor policy, not by the pattern name.** Sub-agents make isolation *possible*; the code supervisor makes it *real* by building a scoped payload for each sub-agent call.

#### 7.2.2 Workflow Topology

```
┌──────────────────────────────────────────────────────────────┐
│              Code Supervisor (orchestrator.py)              │
│         state machine · context filter · metrics log          │
│                    (no LLM calls)                           │
└──────────────────────────┬───────────────────────────────────┘
                           │ fixed workflow
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
   │ Retriever   │  │  Reasoner    │  │  Verifier    │
   │ sub-agent   │─►│  sub-agent   │─►│  sub-agent   │
   │  (LLM)      │  │  (LLM)       │  │  (LLM)       │
   └─────────────┘  └──────┬───────┘  └──────┬───────┘
                           ▲                  │
                           │  handoff (retry) │
                           └──────────────────┘
                              if verifier fails
                              (max 1 retry)
```

**Workflow steps (deterministic):**

| Step | Actor | Action |
|---|---|---|
| 1 | Supervisor | Load question + 10 paragraphs; initialize run state |
| 2 | Retriever sub-agent | Rank paragraphs; return top-k indices + brief rationale |
| 3 | Supervisor | Filter context to top-k paragraphs only; pass to reasoner |
| 4 | Reasoner sub-agent | Multi-hop reasoning; return candidate answer + cited paragraph/sentence indices |
| 5 | Supervisor | Build verifier payload: question + answer + cited facts only (no full paragraphs) |
| 6 | Verifier sub-agent | Validate grounding and consistency; return `pass` or `fail` + feedback |
| 7a | Supervisor (pass) | Emit final answer; write traces |
| 7b | Supervisor (fail) | Hand off feedback to reasoner — **one retry only** (step 4 repeat) |
| 8 | Supervisor | Emit final answer (or best attempt after retry); write traces |

#### 7.2.3 Sub-Agent Roles

| Sub-agent | LLM? | Responsibility |
|---|---|---|
| **Retriever** | Yes | Select top-k relevant paragraphs from the 10 provided |
| **Reasoner** | Yes | Perform multi-hop reasoning over supervisor-filtered paragraphs |
| **Verifier** | Yes | Check answer is grounded in cited facts; flag unsupported claims |
| **Supervisor** | **No** | Run workflow, filter context between steps, enforce `max_rounds`, aggregate metrics |

#### 7.2.4 Context Isolation Policy

The supervisor builds a **different prompt payload** for each sub-agent. This is the core mechanism behind §2.1.

| Sub-agent | Sees | Does NOT see |
|---|---|---|
| **Retriever** | Question + all 10 paragraphs (titles + sentences) | Other agents' reasoning traces |
| **Reasoner** | Question + top-k paragraphs selected by retriever | The other (10−k) distractor paragraphs; verifier feedback (unless retry) |
| **Verifier** | Question + candidate answer + cited facts (indexed sentences) | Full paragraph bodies beyond cited sentences |

Supervisor pseudocode:

```python
# Step 2 — retriever gets everything (must search across all 10)
retriever_input = {"question": q, "paragraphs": all_10}

# Step 3 — supervisor filters before reasoner call
selected = retriever_output["top_k_indices"]
reasoner_input = {"question": q, "paragraphs": [all_10[i] for i in selected]}

# Step 5 — supervisor strips down for verifier
verifier_input = {
    "question": q,
    "answer": reasoner_output["answer"],
    "cited_facts": reasoner_output["citations"],  # title + sentence only
}
```

#### 7.2.5 Supervisor Responsibilities

| Responsibility | Implementation |
|---|---|
| Run fixed workflow | State machine: `retrieve → reason → verify → [retry?] → done` |
| Filter context | Build per-agent payloads per §7.2.4 |
| Enforce turn limits | `max_rounds` default 6 (including retries) |
| Handle handoff | Pass verifier feedback to reasoner on fail (max 1 retry) |
| Log metrics | Per-step latency, input/output tokens, LLM call count |
| Emit final answer | Parse `FINAL ANSWER: ...` from last successful step |

#### 7.2.6 Constraints (Fairness)

- Multi-agent system gets the **same raw context** as single-agent (no extra documents).
- Context isolation is **supervisor-enforced** — sub-agents never receive paragraphs outside their scoped payload.
- Supervisor adds **zero LLM tokens**; its wall-clock time is logged as orchestration overhead.
- Total LLM calls capped at `max_rounds` (default: 6).
- All sub-agents use the **same base model** as the single-agent baseline.

### 7.3 Shared Infrastructure

| Component | Requirement |
|---|---|
| LLM client | Unified wrapper with token counting hooks |
| Prompt templates | Versioned in `prompts/` |
| Config | YAML/JSON per experiment run |
| Logging | Structured JSONL per question |
| Scoring | Shared evaluation module |

---

## 8. Metrics

### 8.1 Quality Metrics

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

**Reliability metrics** (to test §2.2 claims):

| Metric | Description |
|---|---|
| **Hallucination rate** | % of answers citing facts/entities not present in any provided paragraph |
| **Mis-citation rate** | % of answers referencing wrong paragraph indices vs gold `supporting_facts` |
| **Supporting Facts EM/F1** | Official HotpotQA supporting-facts match (optional v1.1, primary for mis-citation) |

**Optional (v1.1):** Joint EM/F1 (answer + supporting facts both correct).

### 8.2 Latency Metrics

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

### 8.3 Token & Cost Metrics

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

### 8.4 Efficiency Ratios (Derived)

| Ratio | Formula | Insight |
|---|---|---|
| **EM per 1k tokens** | `EM / (total_tokens / 1000)` | Quality efficiency |
| **EM per second** | `EM / latency_seconds` | Speed-quality tradeoff |
| **Cost per correct answer** | `total_cost / correct_count` | Economic efficiency |

---

## 9. Experiment Design

### 9.1 Controlled Variables

| Variable | Control |
|---|---|
| Dataset subset | Identical `subset_ids.json` for both architectures |
| Base model | Same model + version |
| Temperature | Fixed (recommend `0` for reproducibility) |
| Context formatting | Shared formatter |
| Scoring | Shared evaluator |
| Random seed | Fixed and logged |

### 9.2 Independent Variable

- `architecture`: `single` | `multi`

### 9.3 Runs

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

### 9.4 Prediction Format

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

### 9.5 Comparison Report

`comparison.md` must answer:

1. Which architecture has higher EM / F1?
2. Which produces **less hallucination** and **fewer mis-citations**?
3. **Delay.** How much slower is multi-agent (mean, p50, p95), where does the time go (per stage and orchestration overhead), and what is the latency ratio? Explain the cause, not only the ranking (§2.3).
4. **Cost.** How many more tokens and how much more USD does multi-agent use, which stage spends them, and what is the cost per correct answer and EM per 1k tokens? Explain the cause (§2.3).
5. **Verdict.** Do the quality and reliability gains pay for the extra delay and cost? If they do not, say so and prefer the single agent for this task.
6. Where does multi-agent win or lose (by type/level)?
7. **Scope.** In a short note, name other situations where multi-agent can help (parallel independent work, a fresh-context reviewer, per-role models or permissions, background work — §2.6). Then state that this test repo only tries to prove **Point 1** (scoped context, §2.1), and that the numbers above are evidence only for that claim, plus the delay and cost of the split (§2.3).
8. **When to keep a single agent.** Name the four situations in §2.7. Tie the "prefer single agent" verdict only to the two this run can speak to: the task is small (`easy` vs `hard` / `bridge`, H6) and coordination costs more than the work (the delay and token bill, §2.3). Do not present tightly linked coding steps, or agents conflicting on the same files, as findings.

---

## 10. System Architecture

```
single-agent-vs-multiple-agent/
├── PRD.md
├── README.md
├── configs/
│   ├── experiment.yaml           # model, subset size, architecture toggles
│   ├── multi_agent.yaml          # top_k, max_rounds, retry policy
│   └── pricing.yaml              # per-model token pricing
├── data/
│   └── subsets/
│       ├── validation_50.json
│       └── validation_500.json
├── src/
│   ├── dataset/
│   │   └── hotpot.py             # load & format HotpotQA
│   ├── agents/
│   │   ├── single.py             # single-agent baseline (§7.1)
│   │   ├── base.py               # shared sub-agent interface
│   │   └── multi/
│   │       ├── orchestrator.py   # code-based supervisor (no LLM)
│   │       ├── workflow.py       # state machine: retrieve → reason → verify → retry
│   │       ├── context_filter.py # context isolation policy (§7.2.4)
│   │       ├── types.py          # workflow state & sub-agent I/O types
│   │       ├── retriever.py      # retriever sub-agent (LLM)
│   │       ├── reasoner.py       # reasoner sub-agent (LLM)
│   │       └── verifier.py       # verifier sub-agent (LLM)
│   ├── llm/
│   │   └── client.py             # unified LLM client + token tracking
│   ├── eval/
│   │   ├── metrics.py            # EM, F1 normalization
│   │   ├── reliability.py        # hallucination & mis-citation rates
│   │   └── compare.py            # generate comparison report
│   ├── utils/
│   │   ├── io.py                 # config loading, JSONL I/O
│   │   └── tracing.py            # per-step workflow trace logging
│   └── runner/
│       └── experiment.py         # CLI entry point
├── prompts/
│   ├── single_agent.txt
│   ├── retriever.txt
│   ├── reasoner.txt
│   ├── reasoner_retry.txt        # reasoner prompt after verifier handoff
│   ├── verifier.txt
│   └── orchestrator.txt          # unused in v1 (supervisor is code)
├── scripts/
│   └── generate_subsets.py
├── results/                      # gitignored experiment outputs
└── notebooks/
    └── analysis.ipynb
```

### 10.1 CLI (target)

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

## 11. Success Criteria

### 11.1 MVP (v1.0)

- [ ] Load HotpotQA distractor validation split from Hugging Face
- [ ] Single-agent solver runnable on a 50-question smoke set
- [ ] Multi-agent custom workflow runnable: code supervisor + retriever/reasoner/verifier sub-agents
- [ ] Supervisor context isolation verified: reasoner payload excludes non-selected paragraphs
- [ ] EM and F1 computed with official normalization
- [ ] Per-question latency and token counts logged
- [ ] Side-by-side `comparison.json` and `comparison.md` generated
- [ ] README with setup and reproduction steps

### 11.2 Benchmark Complete (v1.1)

- [ ] Full 7,405 validation run for both architectures
- [ ] Breakdown by type and level
- [ ] Per-agent token/latency breakdown for multi-agent
- [ ] At least 2 models tested (e.g. GPT-4o + one cheaper model)
- [ ] Final report explains the delay and cost trade-off and whether the quality gain is worth it (§2.3, §9.5)

### 11.3 Hypotheses to Test

Aligned with **§2 What We Want to Prove**. All of H1–H7 test **Point 1 only** (scoped context), plus the delay and cost of that split. They do not test parallel speedup, fresh-context review as its own effect, per-role models or permissions, or background work (§2.6). They also do not test the "keep a single agent" rules in §2.7, except that H6 and the delay/cost verdict speak to a small task that already fits in one context, and to coordination cost.

| # | Hypothesis | Maps to |
|---|---|---|
| H1 | Multi-agent achieves **higher EM/F1**, especially on `hard` / `bridge` questions, because each agent sees filtered context | §2.2 Higher accuracy |
| H2 | Multi-agent produces **fewer hallucinations** — answers not grounded in provided paragraphs | §2.2 Less hallucination |
| H3 | Multi-agent produces **fewer mis-citations** — wrong supporting paragraph attribution | §2.2 Fewer mis-citations |
| H4 | Multi-agent has **≥2× latency** vs single-agent due to orchestration and extra LLM rounds | §2.3 Slower |
| H5 | Multi-agent consumes **≥1.5× tokens** due to inter-agent message passing | §2.3 Higher token cost |
| H6 | Single-agent is more **cost-efficient per correct answer** on `easy` questions where context filtering adds little value | §2.4 Cost per correct answer; §2.7 the task is small |
| H7 | Multi-agent advantage **diminishes** when the base model is stronger (better at ignoring distractors) | §2.5 Null hypothesis |

---

## 12. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| API cost on full validation set | High spend | Staged subsets; cache LLM responses |
| Non-deterministic LLM outputs | Irreproducible results | `temperature=0`; log raw completions; optional rerun |
| Multi-agent prompt bloat | Unfair token comparison | Cap context per agent; log prompt sizes |
| Rate limits | Failed runs | Exponential backoff; resume from checkpoint |
| Answer extraction errors | Scoring noise | Strict output format (`FINAL ANSWER: ...`); parser tests |
| Overfitting prompts to subset | Misleading conclusions | Hold out a separate 500-ID subset for final eval |

---

## 13. Timeline (Suggested)

| Phase | Duration | Deliverable |
|---|---|---|
| **Phase 0 — Setup** | 2–3 days | Dataset loader, LLM client, eval metrics |
| **Phase 1 — Single agent** | 2–3 days | Working single-agent + smoke test results |
| **Phase 2 — Multi agent** | 4–5 days | Code supervisor + 3 sub-agents (custom workflow) |
| **Phase 3 — Pilot** | 2 days | 500-question comparison report |
| **Phase 4 — Full benchmark** | 2–3 days | Full validation run + analysis notebook |
| **Phase 5 — Write-up** | 1–2 days | Final report with recommendations |

**Total estimate:** ~2–3 weeks

---

## 14. Open Questions

1. **Which base model(s)?** Start with one capable model (GPT-4o or Claude Sonnet) + one budget model for cost sensitivity analysis.
2. **Retriever strategy:** LLM-based paragraph ranking vs embedding similarity — affects fairness if only multi-agent gets embeddings.
3. **Verifier loop:** Allow one retry on verification failure, or single-pass only?
4. **Answer format:** Free-form vs forced `FINAL ANSWER: <short phrase>` — recommend forced format for reliable EM.
5. **Parallelism:** Run questions concurrently (faster wall-clock) vs sequentially (cleaner latency per question)?

---

## 15. Appendix

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
| **Orchestrator** | Agent that routes tasks and aggregates outputs (v1: implemented as **code supervisor**, not LLM) |
| **Code supervisor** | Python state machine that runs the custom workflow, filters context, and invokes sub-agents |
| **Sub-agent** | A specialized LLM call with a task-specific prompt and scoped context payload |
| **Custom workflow** | Fixed pipeline (retrieve → reason → verify) with deterministic step order |
| **Handoff** | Supervisor passes verifier feedback back to reasoner for one retry |
| **Round-trip** | One LLM request + response pair |
