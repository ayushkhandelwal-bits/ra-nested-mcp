# What This Project Actually Does (Plain-Language Explanation)

## The problem, in everyday terms

Imagine a phone company. Every time someone makes a call or uses data, the
company is supposed to record it, calculate the price, and bill the
customer. But the internal systems that do this sometimes fail somewhere
along the way — a call happens, but it never gets billed correctly. That
"missing money" is called **revenue leakage**: the company did the work but
never got paid for some of it. Left unmonitored, this adds up to real,
significant financial loss. Companies have entire teams — Revenue Assurance
(RA) teams — whose job is to catch and explain this leakage.

## What I built, in one sentence

A smart assistant for that team: instead of manually reading documentation,
running spreadsheets, and building forecasts separately, a person can ask
one plain-English question and get back a complete, data-backed answer.

## The three kinds of questions it can answer

1. **"Why does this normally happen?"** — like someone who has read every
   internal manual and can explain typical causes (a mediation system
   crashed, a billing bug caused duplicate records, etc.).
2. **"What does the actual data say happened?"** — like a very fast
   accountant who can instantly scan months of records and report: how
   much money was lost, which days were unusually bad, which region or
   service had the worst problems, and whether a bad day was a real issue
   or just normal variation.
3. **"What's likely to happen next?"** — like a weather forecaster, but for
   money: predicting expected leakage over the next 30 days based on
   historical patterns.

Ask something like *"Why was leakage high in June, and what should we
expect next month?"* and the system figures out on its own which of the
above it needs, goes and gets each answer, and writes one clear response —
the way a person would write a report — rather than requiring you to run
three separate tools yourself.

## How it's built, in one sentence

Four small, specialized programs: one that knows the documentation, one
that crunches the numbers, one that forecasts, and one "manager" program
that takes the question, decides which of the other three to consult, and
combines their answers into a single response. (For the technical
architecture — the actual MCP server layout, ports, and tool names — see
`README.md`.)

---

## How the forecasting actually works

1. **Find the pattern.** Daily leakage isn't random — it usually has a
   repeating weekly rhythm (e.g. weekends behave differently from
   weekdays). The system separates history into: the overall trend (rising
   or falling over months), the weekly repeating shape, and leftover
   random noise.
2. **Project the pattern forward.** A well-established forecasting method
   (Holt-Winters) combines the trend and the weekly shape and extends both
   into the next 30 days — e.g. "leakage has been slowly rising each month,
   AND Mondays are always worse than Thursdays, so combine both facts to
   predict each of the next 30 days."
3. **Test itself honestly first.** Before trusting the forecast, it hides
   the most recent real days, pretends not to know them, forecasts them
   anyway, and compares the guess to what actually happened. This test is
   always done in real time order (past → future) — never a random
   shuffle, since shuffling would let it "cheat" by learning from the
   future.
4. **Give a range, not a false-confidence single number.** Instead of "you
   will lose exactly ₹50,000," it simulates many slightly different
   possible futures (500 simulated paths) and reports a realistic range
   (e.g. ₹40,000–₹60,000) alongside its single best guess.

**In one line:** it learns the trend and weekly rhythm from real history,
honestly tests itself on data it hasn't "seen," and reports an honest range
instead of fake precision.

## How anomaly detection works

The system watches each day's numbers against what's "normal" for the
recent weeks around it (a rolling baseline), and flags any day that's
statistically far outside that normal range — not just "different," but
different enough that it's unlikely to be random noise. It supports two
methods: one based on how many standard deviations a value is from the
rolling average, and one based on quartile-range outlier detection — both
standard, well-understood statistical techniques, not an arbitrary
hard-coded threshold.

## How the hypothesis testing works

When comparing "before an incident" vs "during/after an incident," the
system doesn't just eyeball the difference — it runs a real statistical
test to check whether the difference could just be random chance. It's
smart about *which* test to use: it checks whether the data looks
normally distributed and how much data there is, and automatically picks
the appropriate test (a more powerful test when the data allows it, a
safer non-parametric test when it doesn't). Crucially, it separates two
different questions that are often confused: *is this difference
statistically real* (unlikely to be chance) and *is this difference big
enough to matter to the business* (e.g. a real but tiny 1% change usually
isn't worth acting on operationally, even if statistically "real").

## Why it's designed this way (the engineering rationale)

- **Separation of concerns:** each capability (knowledge lookup,
  statistics, forecasting) is its own small, independently testable
  service, not one giant program.
- **Extended, not rebuilt:** the analytics and forecasting layers were
  added on top of an already-working system (the original knowledge
  agent) without changing or breaking what already worked.
- **Numbers come from real code, not the AI's guesswork:** every number
  reported comes from a deterministic calculation (pandas/scipy/
  statsmodels), not from the language model doing arithmetic in its head.
  The model's only job is deciding which tool to use and explaining the
  result in plain language — this keeps every number auditable and
  reproducible.
- **Real analytics discipline, not just "ran a library once":**
  auto-selecting the right statistical test, explicitly separating
  statistical significance from business significance, calling out that
  correlation isn't causation, and using time-respecting (not shuffled)
  splits for forecasting — the kind of care that distinguishes doing
  analysis correctly from just running a function and hoping.

## Quick interview cheat-sheet: known discrepancies to expect

- The resume bullet mentions "Llama 3.3 70B" as the underlying model.
  During live testing, that exact model wasn't available on the account's
  Groq access, so the system currently runs on `openai/gpt-oss-120b`
  instead (confirmed via Groq's own model-listing API). The architecture
  is model-agnostic — swapping the underlying LLM required no design
  changes, only a config value — so this is a reasonable, honest point to
  make if asked.