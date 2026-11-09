
import json
import time
import uuid
from pathlib import Path

import mock_llm
import importlib.util


# --------------------------------------------------
# 1. PROJECT PATHS AND TOOL IMPORTS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = BASE_DIR / "state"
OUTPUT_DIR = BASE_DIR / "outputs"

STATE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

tools_path = Path(__file__).resolve().parent / "19_agent_tools.py"
spec = importlib.util.spec_from_file_location("agent_tools", tools_path)
agent_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_tools)

TOOLS_SPEC = agent_tools.TOOLS_SPEC
TOOL_MAP = agent_tools.TOOL_MAP

MAX_ITERATIONS = 6
TOKEN_LIMIT = 6000
TIME_LIMIT_SECONDS = 10


# --------------------------------------------------
# 2. CHECKPOINT AND LOGGING
# --------------------------------------------------

def save_checkpoint(state):
    path = STATE_DIR / f"{state['run_id']}.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=4)

    return path


def load_checkpoint(run_id):
    path = STATE_DIR / f"{run_id}.json"

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {run_id}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def log_agent_run(result):
    path = OUTPUT_DIR / "agent_runs.jsonl"

    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(result) + "\n")


# --------------------------------------------------
# 3. TOOL EXECUTION
# --------------------------------------------------

def execute_tool(tool_name, arguments):
    if tool_name not in TOOL_MAP:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        return TOOL_MAP[tool_name](**arguments)
    except Exception as error:
        return {"error": str(error)}


# --------------------------------------------------
# 4. FALLBACK
# --------------------------------------------------

def make_fallback(status, steps):
    return {
        "status": status,
        "final_answer": (
            "FALLBACK: Processing stopped safely. "
            "Please send this request to a human support agent."
        ),
        "steps": steps,
    }


# --------------------------------------------------
# 5. REACT AGENT LOOP
# --------------------------------------------------

def continue_agent(
    state,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tool_executions=None
):

    start_time = time.monotonic()
    tool_executions = len(state["tools_executed"])

    result = None

    remaining = max_iterations - state["llm_calls"]

    for _ in range(max(0, remaining)):

        # Time limit
        if time.monotonic() - start_time >= time_limit_seconds:
            result = make_fallback(
                "FALLBACK_TIMEOUT", state["steps"]
            )
            break

        # Call mock LLM
        response = mock_llm.chat(
            state["messages"],
            tools=TOOLS_SPEC
        )

        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens

        # Token limit: do not execute any requested tool
        if state["tokens_used"] > token_limit:
            result = make_fallback(
                "FALLBACK_TOKEN_LIMIT", state["steps"]
            )
            break

        message = response.choices[0].message
        tool_calls = message.tool_calls

        # Final answer
        if not tool_calls:
            final_answer = message.content or "No answer returned."
            status = final_answer.split(":")[0].strip()

            result = {
                "status": status,
                "final_answer": final_answer,
                "steps": state["steps"],
            }
            break

        # Record assistant tool calls
        assistant_calls = []

        for call in tool_calls:
            assistant_calls.append({
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_calls,
        })

        # Execute requested tools
        for call in tool_calls:

            tool_name = call.name
            arguments = call.arguments

            tool_result = execute_tool(
                tool_name,
                arguments
            )

            state["tools_executed"].append(tool_name)
            tool_executions += 1

            state["steps"].append({
                "tool": tool_name,
                "arguments": arguments,
                "result": tool_result,
            })

            # Append tool observation
            state["messages"].append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(tool_result),
            })

            state["step_number"] += 1

            # Save checkpoint after every tool
            save_checkpoint(state)

            # Simulate crash for resume demonstration
            if (
                crash_after_tool_executions is not None
                and tool_executions >= crash_after_tool_executions
            ):
                return {
                    "status": "INTERRUPTED",
                    "final_answer": "Run interrupted after checkpoint.",
                    "steps": state["steps"],
                    "run_id": state["run_id"],
                    "llm_calls": state["llm_calls"],
                    "tokens_used": state["tokens_used"],
                }

    if result is None:
        result = make_fallback(
            "FALLBACK_MAX_ITERATIONS", state["steps"]
        )

    result.update({
        "scenario_id": state["scenario_id"],
        "run_id": state["run_id"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "tools_executed": state["tools_executed"],
    })

    log_agent_run(result)
    save_checkpoint(state)

    return result


# --------------------------------------------------
# 6. START A NEW AGENT RUN
# --------------------------------------------------

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tool_executions=None
):

    mock_llm.reset()

    run_id = str(uuid.uuid4())

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,

        "messages": [
            {
                "role": "system",
                "content": (
                    "AGENT: You are an Instamart refund support agent."
                ),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],

        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
    }

    save_checkpoint(state)

    return continue_agent(
        state,
        max_iterations,
        token_limit,
        time_limit_seconds,
        crash_after_tool_executions,
    )


# --------------------------------------------------
# 7. RESUME AN INTERRUPTED RUN
# --------------------------------------------------

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):

    state = load_checkpoint(run_id)

    return continue_agent(
        state,
        max_iterations,
        token_limit,
        time_limit_seconds,
    )


# --------------------------------------------------
# 8. BASIC TEST
# --------------------------------------------------

if __name__ == "__main__":

    import csv

    scenario_file = BASE_DIR / "data" / "raw" / "agent_scenarios.csv"

    with open(scenario_file, "r", encoding="utf-8") as file:
        scenarios = list(csv.DictReader(file))

    print("\n--- SEVEN SCENARIO TESTS ---")

    for scenario in scenarios:
        result = run_agent(
            scenario["message"],
            scenario["scenario_id"]
        )

        print(
            scenario["scenario_id"],
            "Expected:", scenario["expected_outcome"],
            "Actual:", result["status"],
            "Calls:", result["llm_calls"],
        )

    # Checkpoint recovery demonstration
    print("\n--- CHECKPOINT RECOVERY TEST ---")

    s1 = scenarios[0]

    interrupted = run_agent(
        s1["message"],
        "S1_RESUME_DEMO",
        crash_after_tool_executions=2
    )

    print("Initial status:", interrupted["status"])
    print("Saved run ID:", interrupted["run_id"])

    resumed = resume_agent(interrupted["run_id"])

    print("Resumed status:", resumed["status"])
    print("Final answer:", resumed["final_answer"])
    print("Tools executed:", resumed["tools_executed"])

    # Token limit demonstration
    print("\n--- TOKEN LIMIT TEST ---")

    token_test = run_agent(
        s1["message"],
        "TOKEN_LIMIT_TEST",
        token_limit=200
    )

    print("Token test status:", token_test["status"])
    print("Tokens used:", token_test["tokens_used"])

    # Timeout demonstration
    print("\n--- TIME LIMIT TEST ---")

    timeout_test = run_agent(
        s1["message"],
        "TIME_LIMIT_TEST",
        time_limit_seconds=0
    )

    print("Timeout status:", timeout_test["status"])

    print("\nAll agent demonstrations completed.")