# Context Compaction Strategies

Demonstrates how context compaction strategies affect agent performance on a multi-step code review task.

## Running

Open `compaction.ipynb` in Jupyter and run all cells. Requires:

- Azure OpenAI API access (set `AZURE_OPENAI_ENDPOINT` in `picoagents/.env`)
- PicoAgents installed: `pip install -e ".[all]"` from `picoagents/`

Each run takes several minutes (5 agent runs with real API calls).

## What It Shows

The notebook runs an exhaustive code review of the [handtracking](https://github.com/victordibia/handtracking) repository (~44 Python files) with five compaction configurations:

| Run | Strategy | Budget | Hook | What it tests |
|-----|----------|--------|------|---------------|
| NoCompaction | None | — | Yes | Baseline — unbounded context growth |
| HeadTail 8k | HeadTail | 8,000 | Yes | Aggressive compaction + hook interaction |
| HeadTail 8k (no hook) | HeadTail | 8,000 | No | Isolates completion hook effect |
| HeadTail 15k | HeadTail | 15,000 | Yes | Moderate compaction budget |
| Isolation | Sub-agent delegation | 50,000 | Yes | Coordinator + sub-agent pattern |

Key outputs:
- **Context growth chart** — API-reported input tokens per LLM call showing sawtooth compaction patterns
- **Trace analysis** — tool call batching, file redundancy, and thrashing detection
- **Cost vs quality scatter** — total tokens and latency vs LLM-as-judge quality scores
- **Judge reasoning** — per-criterion evaluation explaining *why* each strategy scored as it did

Results are persisted via `EvalResults` — re-run visualizations without re-running agents.

## Key Insights

**Compaction reallocates tokens, it doesn't necessarily reduce them.** An agent with compaction may run more iterations (more LLM calls), each with a smaller context. Total tokens can be similar to no-compaction, but each iteration is more productive because the agent isn't dragging along verbatim history of every prior file read.

**Thrashing is the failure mode.** When the budget is too tight, the agent reads files, compaction drops them, and the agent re-reads the same files. The signal: high duplicate read ratios and a flat token line clamped at the budget. In this experiment, HeadTail 8k showed 55% redundant reads vs 15% for the no-hook variant.

**Completion hooks and compaction interact.** Without a hook, the agent stops the first time it decides it's done. With a hook, the agent gets pushed back to keep working. Whether this helps depends on whether the extra work is productive or just more thrashing at a tight budget.

**Budget sizing rule of thumb:** set the budget to 2-3x the typical working set. Below peak context = active compaction; above = no effect. The sweet spot is where compaction fires intermittently but duplicate reads stay low.

## When to Use What

| Scenario | Strategy | Budget |
|----------|----------|--------|
| Multi-step tool tasks | `HeadTailCompaction` | 2-3x typical working set |
| Short tasks (< 5 steps) | `NoCompaction` | N/A |
| Debugging / benchmarking | `NoCompaction` | N/A |

## Further Reading

This notebook is the basis for the blog post [Context Engineering 101: How Agents Use LLMs](https://newsletter.victordibia.com/p/context-engineering-101-how-agents), which covers compaction, isolation, instructions, and tool design in broader context.
