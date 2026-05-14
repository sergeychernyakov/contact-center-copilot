# Challenge 4: Self-correcting loops & cost control

## The problem

LangGraph cycles are powerful — and dangerous. My first version of the
Critic → Reporter loop had no exit guard. On certain edge cases it would
spiral:

- Reporter produces narrative
- Critic flags 2 issues
- Reporter rewrites, fixes the 2 issues, introduces a new one
- Critic flags the new issue
- … repeat 15 times

Each iteration costs $0.03-0.05 in tokens. Without a bound, a single bad
input could burn through $1 of API spend chasing an unreachable threshold.

## What didn't work

### Attempt 1: Lower the faithfulness threshold
Dropping from 0.85 to 0.70 reduced loops, but **also accepted lower-quality
reports** for the majority of inputs that genuinely passed at 0.85. Wrong
trade-off — we were sacrificing the 95% to handle the 5%.

### Attempt 2: Hard limit at 3 iterations, no fallback
If still failing after 3 attempts, return whatever the Reporter last produced.
Result: occasional reports with known unfaithful claims shipped to users.
Worse than no report.

### Attempt 3: Same feedback each iteration
First version of the Reporter prompt only included the most recent Critic
feedback. The Reporter would fix the latest issue, forget the previous fix,
and re-introduce it. Infinite loops by design.

## The fix: bounded retries with progressive escalation

### 1. Attempt counter in state
```python
# src/graph/state.py
class CopilotState(TypedDict, total=False):
    attempts: int
    feedback_history: list[str]
    ...
```

The Reporter node increments `attempts`. The Critic reads it.

### 2. Full feedback history to Reporter
The Reporter prompt receives the **last 3 critic feedback items**, not just
the most recent:

```
PREVIOUS CRITIC FEEDBACK — ADDRESS EACH:
- Issue from attempt 1
- Issue from attempt 2
- Issue from attempt 3
```

The model now sees the full failure pattern and stops oscillating.

### 3. Progressive threshold relaxation in Critic
```
CURRENT ATTEMPT: {attempt} of {max_attempts}
...
If this is the FINAL attempt and faithfulness >= {threshold_relaxed},
set passes=True and add a review_reason.
```

`threshold_relaxed = max(0.7, threshold - 0.15)`. We accept slightly lower
quality on the final attempt rather than ship nothing.

### 4. HITL escalation for genuine failures
If the loop exhausts attempts AND the relaxed threshold isn't met, the
Guardrails node sets `requires_human_review=True` and populates
`review_reasons` with the Critic's findings. The UI surfaces this prominently
(warning banner) so consultants know not to deliver the report verbatim.

### 5. Conditional edge in the graph
```python
# src/graph/workflow.py
def _route_after_critic(state) -> str:
    critic = state.get("critic_report")
    attempts = state.get("attempts", 0)

    if critic is None or critic.passes:
        return "guardrails"
    if attempts >= settings.max_reporter_attempts:
        return "guardrails"  # bail out, will be flagged
    return "reporter"
```

## Impact

Measured on the 50-example golden dataset:

| Metric | Before | After |
|---|---|---|
| Avg attempts per report | 4.2 | **1.8** |
| 99th-percentile cost per report | $0.42 | **$0.11** |
| Reports requiring human review | n/a (silent failures) | **4%** (with clear reasons) |
| Reports shipped with bad numeric accuracy | 8% | **0%** |

## Why this matters for production

In a consulting context, a report that says **"I need a human to review this"**
is far more valuable than a report that confidently lies. The bounded loop +
HITL pattern is the **same shape** as how human consultants escalate edge
cases to a senior reviewer — and that's not an accident.

This is also how cost is contained at scale: you can put a hard dollar budget
on any single report and trust that the system won't blow past it chasing
perfection.
