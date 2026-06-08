# korrel

OSS Python SDK for agent simulation. Define a multi-turn agent test once, run it as a pytest CI gate, export it as a verifiers/OpenEnv RL environment.

This is the dispatch A core: the package skeleton, the four data objects, and the runtime simulation loop, delivered as an importable Python API. The `korrel run` CLI, the pytest plugin, and telemetry arrive in dispatch B.

## Install

```
uv add korrel
```

Bring your own provider keys. Korrel reads keys from the environment at call time and stores none. The default provider is Claude via the `anthropic` SDK; set `ANTHROPIC_API_KEY`. OpenAI support is an optional extra (`korrel[openai]`).

## The four objects

- `Scenario`: a code-first definition of a test. Holds the system prompt, a `Persona`, the opening message, the mock tools, `max_turns`, a `seed`, ground-truth `info`, and a `Rubric`.
- `Persona`: the LLM-driven user-simulator. Given the conversation so far, it produces the next user message. Defaults to Claude.
- `MockTool`: a programmable tool. Holds a chat-completions tool schema and a `respond` callable that takes the parsed arguments and a mutable per-run state and returns a result.
- `Rubric`: reward functions plus an optional hardened LLM judge. Reward signatures mirror verifiers, `(completion, info, **kwargs) -> float`. The judge treats the transcript as data, never as instructions.

## Sketch

```python
from korrel import MockTool, Persona, Rubric, Scenario, Message, run_scenario

def lookup_order(args, state):
    state["called"] = True
    return {"order_id": args["order_id"], "amount": 49.99, "refundable": True}

orders = MockTool(
    name="lookup_order",
    schema={"type": "function", "function": {
        "name": "lookup_order",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
    }},
    respond=lookup_order,
)

def confirmed(completion, info):
    return 1.0 if any(m.role == "assistant" and "refund" in (m.content or "").lower()
                      for m in completion) else 0.0

scenario = Scenario(
    id="support_refund",
    system="You are a support agent. Resolve refunds with lookup_order.",
    persona=Persona(goal="Get a refund for order A1001.", behavior="Polite but firm."),
    opening_message="My order A1001 arrived broken and I want a refund.",
    tools=[orders],
    max_turns=2,
    seed=7,
    info={"order_id": "A1001"},
    rubric=Rubric(funcs=[confirmed], pass_threshold=0.5),
)

# adapter is the agent under test: a callable (messages, tools) -> assistant Message.
result = run_scenario(scenario, my_adapter)
print(result.score, result.passed, result.failed_functions)
```

`examples/support_refund.py` holds a runnable scenario definition.

## Determinism

Every run takes a seed and records the model and request parameters. The seed pins scenario setup and any sampling Korrel controls. LLM calls are not bit-reproducible; provider nondeterminism is outside the seed.

## License

MIT.
