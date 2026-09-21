import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Literal

from groq import AsyncGroq

from mcp2_orchestrator.mcp1_client import Mcp1Client
from mcp2_orchestrator.mcp3_client import Mcp3Client
from mcp2_orchestrator.mcp4_client import Mcp4Client
from mcp2_orchestrator.settings import settings

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 12

SYSTEM_PROMPT = """\
You are a Revenue Assurance (telecom) analytics and decision-support agent. \
You have three kinds of tools and MUST only answer from what they return — \
never from prior knowledge or by guessing at numbers:

1. `search_knowledge` — domain knowledge (what RA concepts mean, typical \
   causes of leakage, how CDR/mediation/rating/billing pipelines work). Use \
   this for "what does X mean" / "why does this type of failure cause \
   leakage" questions.
2. Analytics tools (`get_ra_summary`, `get_kpi_metrics`, \
   `get_revenue_leakage_trend`, `detect_anomalies`, `analyze_incident`, \
   `run_hypothesis_test`, `get_correlation_analysis`, `analyze_by_region`, \
   `analyze_by_service`, `analyze_by_network`, `analyze_by_plan`) — the \
   actual operational data: what happened, whether it's statistically \
   unusual, which operational incident coincided with it, which segment is \
   driving it.
3. Forecasting tools (`forecast_revenue_leakage`, `compare_forecast_models`, \
   `get_forecast_accuracy`, `get_forecast_intervals`) — what's likely to \
   happen next. Only call these if the question actually asks about the \
   future / a forecast; don't call them for purely descriptive or \
   diagnostic questions.

You maintain a scratchpad of tasks to track your research progress. Before \
doing anything, always consult the current scratchpad state provided below.

Your workflow:
1. Decompose the question into tasks using `add_task`. Use `depends_on` when one task requires
   the result of another (e.g. "check for an incident on that date" depends on "find the anomalous date").
2. Execute tasks by calling the minimum set of tools necessary — do not call every tool for every
   question. A purely descriptive question ("what was leakage last month") only needs an analytics
   tool. A purely conceptual question only needs search_knowledge. A "why did X spike and what's next"
   question needs analytics (to find/confirm the anomaly and any incident), search_knowledge (to explain
   the likely mechanism), and a forecast tool (only if the future was actually asked about).
3. Mark each task done with `complete_task` once you have retrieved useful results.
4. When all tasks needed to answer the question are complete, call `finish` with a synthesized answer.

When you write the final answer:
- Clearly separate: observed facts from the data, information retrieved from RA documents,
  statistical inference (e.g. "3.2 standard deviations above baseline", "p=0.01"), model
  predictions/forecasts, and your own business recommendations. Never present a forecast or a
  correlation as an observed fact.
- If you used get_correlation_analysis, explicitly note correlation does not imply causation.
- If you used a forecast tool, explicitly note it is a projection with uncertainty, not a guarantee.
- Where useful for a diagnostic or combined question, structure the answer with short sections:
  Executive Summary, Key Findings, Statistical Evidence, Likely Drivers, Relevant Domain Knowledge,
  Forecast (if applicable), Recommended Actions, Limitations. Skip sections that don't apply — a
  simple descriptive question doesn't need all of these.

Rules:
- Always use the tools — never answer from memory, and never fabricate a statistic, date, or figure.
- If a tool call returns an "error" field, that's a real business-logic error (e.g. an unknown
  event_id, or too little data for a holdout test) — do not treat its message as an answer; adjust
  your arguments and retry, or explain the limitation in your final answer.
- Call `finish` only when you have enough retrieved information to answer fully.
- If you hit a dead end on a task, mark it complete with a note and move on.
"""

# --- Tool schemas, OpenAI-compatible function-calling format (Groq uses this) ---

_DATE_ARGS = {
    "start_date": {"type": "string", "description": "ISO date (YYYY-MM-DD), inclusive. Omit for full history."},
    "end_date": {"type": "string", "description": "ISO date (YYYY-MM-DD), inclusive. Omit for full history."},
}
_METRIC_ENUM = ["revenue_leakage", "failed_cdrs", "mediation_failures", "rating_failures", "billing_mismatches", "processing_delay_avg", "duplicate_cdrs"]

_KNOWLEDGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search the Revenue Assurance knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results (default 3)", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
]

_ANALYTICS_TOOLS = [
    {"type": "function", "function": {
        "name": "get_ra_summary",
        "description": "High-level 'what happened' summary: KPIs, descriptive statistics, and trend direction for revenue leakage over a period.",
        "parameters": {"type": "object", "properties": {**_DATE_ARGS}},
    }},
    {"type": "function", "function": {
        "name": "get_kpi_metrics",
        "description": "Standard RA KPI set (leakage rate, recovery rate, failure rates, etc.) over a date range, with definitions.",
        "parameters": {"type": "object", "properties": {**_DATE_ARGS}},
    }},
    {"type": "function", "function": {
        "name": "get_revenue_leakage_trend",
        "description": "Daily revenue_leakage time series with a rolling mean, for trend analysis.",
        "parameters": {"type": "object", "properties": {**_DATE_ARGS, "rolling_window": {"type": "integer", "default": 7}}},
    }},
    {"type": "function", "function": {
        "name": "detect_anomalies",
        "description": "Detect anomalous days on a chosen metric via z-score (rolling baseline) or IQR. Returns date, observed vs baseline, deviation, anomaly score, severity for each flagged day.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": _METRIC_ENUM, "default": "revenue_leakage"},
            "method": {"type": "string", "enum": ["zscore", "iqr"], "default": "zscore"},
            "window": {"type": "integer", "default": 14},
            "threshold": {"type": "number", "default": 3.0},
            **_DATE_ARGS,
        }},
    }},
    {"type": "function", "function": {
        "name": "analyze_incident",
        "description": "Pre/during/post-incident analysis for a specific operational_events.csv event_id: metric shifts, a pre-vs-during hypothesis test, estimated extra leakage, whether metrics recovered to baseline.",
        "parameters": {"type": "object", "properties": {
            "event_id": {"type": "string", "description": "e.g. 'EVT003'"},
            "pre_days": {"type": "integer", "default": 14},
            "post_days": {"type": "integer", "default": 14},
        }, "required": ["event_id"]},
    }},
    {"type": "function", "function": {
        "name": "run_hypothesis_test",
        "description": "Test whether a metric differs significantly before vs. on/after a split_date. Auto-selects Welch's t-test or Mann-Whitney U. Reports statistical AND business significance separately.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": _METRIC_ENUM, "default": "revenue_leakage"},
            "split_date": {"type": "string", "description": "ISO date to split before/after, e.g. an incident start_date"},
            **_DATE_ARGS,
            "alpha": {"type": "number", "default": 0.05},
        }, "required": ["split_date"]},
    }},
    {"type": "function", "function": {
        "name": "get_correlation_analysis",
        "description": "Pearson and Spearman correlation of operational metrics against revenue_leakage. Correlation, not causation.",
        "parameters": {"type": "object", "properties": {**_DATE_ARGS}},
    }},
    {"type": "function", "function": {
        "name": "analyze_by_region",
        "description": "Revenue leakage and failure counts broken down by region (North/South/East/West).",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "analyze_by_service",
        "description": "Revenue leakage and failure counts broken down by service_type (Voice/SMS/Data/Roaming).",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "analyze_by_network",
        "description": "Revenue leakage and failure counts broken down by network_type (4G/5G) — e.g. checking whether leakage concentrates in 5G.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "analyze_by_plan",
        "description": "Revenue leakage and failure counts broken down by plan_type (Prepaid/Postpaid).",
        "parameters": {"type": "object", "properties": {}},
    }},
]

_FORECAST_TOOLS = [
    {"type": "function", "function": {
        "name": "forecast_revenue_leakage",
        "description": "Forecast daily revenue_leakage N days ahead (Holt-Winters, trend + weekly seasonality) with prediction intervals and a direction/limitations summary. Only call this if the question actually asks about the future.",
        "parameters": {"type": "object", "properties": {
            "steps": {"type": "integer", "default": 30},
            "intervals": {"type": "boolean", "default": True},
        }},
    }},
    {"type": "function", "function": {
        "name": "compare_forecast_models",
        "description": "Compare seasonal-naive baseline vs Holt-Winters on a chronological holdout (MAE/RMSE/MAPE).",
        "parameters": {"type": "object", "properties": {"test_days": {"type": "integer", "default": 30}}},
    }},
    {"type": "function", "function": {
        "name": "get_forecast_accuracy",
        "description": "Day-by-day accuracy for one model over a chronological holdout.",
        "parameters": {"type": "object", "properties": {
            "model": {"type": "string", "enum": ["holt_winters", "seasonal_naive"], "default": "holt_winters"},
            "test_days": {"type": "integer", "default": 30},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_forecast_intervals",
        "description": "Forecast values with 80%/95% prediction intervals for the next N days, no narrative fields.",
        "parameters": {"type": "object", "properties": {"steps": {"type": "integer", "default": 30}}},
    }},
]

_CONTROL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": "Add a new research task to the scratchpad.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "What this task will research"},
                    "depends_on": {"type": "array", "items": {"type": "integer"}, "description": "Task IDs this task must wait for", "default": []},
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark a task as complete and record its result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "ID of the task"},
                    "result": {"type": "string", "description": "Summary of what was found"},
                },
                "required": ["task_id", "result"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Provide the final synthesized answer and end the research loop.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "Final answer based on retrieved information"}},
                "required": ["answer"],
            },
        },
    },
]

TOOLS = _KNOWLEDGE_TOOLS + _ANALYTICS_TOOLS + _FORECAST_TOOLS + _CONTROL_TOOLS

_ANALYTICS_TOOL_NAMES = {t["function"]["name"] for t in _ANALYTICS_TOOLS}
_FORECAST_TOOL_NAMES = {t["function"]["name"] for t in _FORECAST_TOOLS}
_DATA_TOOL_NAMES = {"search_knowledge"} | _ANALYTICS_TOOL_NAMES | _FORECAST_TOOL_NAMES


@dataclass
class Task:
    id: int
    description: str
    status: Literal["pending", "running", "complete", "blocked"] = "pending"
    depends_on: list[int] = field(default_factory=list)
    result: str | None = None


@dataclass
class Scratchpad:
    question: str
    tasks: list[Task] = field(default_factory=list)
    final_answer: str | None = None
    _next_id: int = 0

    def add_task(self, description: str, depends_on: list[int] | None = None) -> int:
        task_id = self._next_id
        self._next_id += 1
        self.tasks.append(Task(id=task_id, description=description, depends_on=depends_on or []))
        return task_id

    def complete_task(self, task_id: int, result: str) -> None:
        for t in self.tasks:
            if t.id == task_id:
                t.status = "complete"
                t.result = result
                return

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "tasks": [
                {"id": t.id, "description": t.description, "status": t.status, "depends_on": t.depends_on, "result": t.result}
                for t in self.tasks
            ],
            "final_answer": self.final_answer,
        }


class Agent:
    def __init__(self, mcp1: Mcp1Client, mcp3: Mcp3Client | None = None, mcp4: Mcp4Client | None = None):
        self._mcp1 = mcp1
        self._mcp3 = mcp3 or Mcp3Client()
        self._mcp4 = mcp4 or Mcp4Client()
        self._llm = AsyncGroq(api_key=settings.groq_api_key)

    async def _dispatch_data_tool(self, tc) -> dict:
        """Route a single data tool call (knowledge / analytics / forecast) to the
        right MCP server and wrap the result as an OpenAI-style tool message."""
        name = tc.function.name
        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        logger.info("Invoking tool: %s  args=%r", name, args)

        try:
            if name == "search_knowledge":
                result = await self._mcp1.search(query=args["query"], top_k=args.get("top_k", 3))
            elif name in _ANALYTICS_TOOL_NAMES:
                result = await self._mcp3.call_tool(name, args)
            elif name in _FORECAST_TOOL_NAMES:
                result = await self._mcp4.call_tool(name, args)
            else:
                result = {"error": f"Unknown data tool '{name}'"}
        except Exception as e:
            logger.warning("Tool call %s failed: %s", name, e)
            result = {"error": f"Tool call failed: {e}"}

        return {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)}

    async def run(self, question: str) -> str:
        logger.info("Agent starting — question: %r", question)
        scratchpad = Scratchpad(question=question)
        messages: list[dict] = [{"role": "user", "content": question}]

        for iteration in range(MAX_ITERATIONS):
            logger.debug("Iteration %d", iteration)
            system_content = SYSTEM_PROMPT + "\n\n--- Current Scratchpad ---\n" + json.dumps(scratchpad.to_dict(), indent=2)

            response = None
            last_error = None
            for attempt in range(3):
                try:
                    response = await self._llm.chat.completions.create(
                        model=settings.groq_model,
                        max_tokens=1536,
                        messages=[{"role": "system", "content": system_content}] + messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        parallel_tool_calls=False,
                    )
                    break
                except Exception as e:  # Groq occasionally emits a malformed
                    # tool call as raw text (tool_use_failed) — retry, it's
                    # usually a one-off generation glitch.
                    last_error = e
                    logger.warning("LLM call failed (attempt %d/3): %s", attempt + 1, e)
                    await asyncio.sleep(1)
            if response is None:
                raise last_error

            choice = response.choices[0].message
            tool_calls = choice.tool_calls or []

            messages.append({
                "role": "assistant",
                "content": choice.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ] if tool_calls else None,
            })

            if not tool_calls:
                if choice.content:
                    return choice.content
                break

            data_calls = [tc for tc in tool_calls if tc.function.name in _DATA_TOOL_NAMES]
            control_calls = [tc for tc in tool_calls if tc.function.name not in _DATA_TOOL_NAMES]

            tool_result_messages: list[dict] = []

            if data_calls:
                data_results = await asyncio.gather(*[self._dispatch_data_tool(tc) for tc in data_calls])
                tool_result_messages.extend(data_results)

            done = False
            for tc in control_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                fn = tc.function.name

                if fn == "add_task":
                    logger.info("Invoking tool: add_task  description=%r  depends_on=%s", args["description"], args.get("depends_on", []))
                    task_id = scratchpad.add_task(description=args["description"], depends_on=args.get("depends_on", []))
                    tool_result_messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"task_id": task_id})})

                elif fn == "complete_task":
                    logger.info("Invoking tool: complete_task  task_id=%s", args["task_id"])
                    scratchpad.complete_task(args["task_id"], args["result"])
                    tool_result_messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"status": "ok"})})

                elif fn == "finish":
                    logger.info("Invoking tool: finish")
                    scratchpad.final_answer = args["answer"]
                    tool_result_messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"status": "done"})})
                    done = True

            messages.extend(tool_result_messages)

            if done:
                break

        if scratchpad.final_answer:
            return scratchpad.final_answer

        completed = [t for t in scratchpad.tasks if t.status == "complete" and t.result]
        if completed:
            partial = "\n".join(f"- {t.description}: {t.result}" for t in completed)
            return f"[Partial answer — max iterations reached]\n{partial}"

        return "Unable to answer: no results retrieved within iteration limit."
