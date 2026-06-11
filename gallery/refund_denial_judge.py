"""Judge-scored rubric, with a deterministic offline path.

Pattern: a Rubric that can score with an LLM judge (``make_judge``). A judge
makes a model call, so it is never run in CI. The offline path scores with a
deterministic reward function (keyword checks), so ``korrel run`` passes with no
key. The judge is constructed below and shown as the live-swap; it is not wired
into the offline scenario.

Offline/needs-key: runs offline with no key on the deterministic reward. The
judge path needs a key.

Live run with the judge (needs a key): score with ``quality_judge`` (see the
swap at the bottom) and set ANTHROPIC_API_KEY. The judge treats the transcript
as untrusted data, never as instructions.

Export: verified against verifiers and openenv (see gallery/index.json). The
exported scenario carries the deterministic rubric, so the generated
environment loads with no key.
"""

from typing import Any

from korrel import Persona, Rubric, Scenario
from korrel.rubric import make_judge
from korrel.types import Message


def explains_denial(
    completion: list[Message], info: dict[str, Any], **kwargs: Any
) -> float:
    """Return 1.0 when an assistant turn gives the reason and a next step.

    Deterministic stand-in for the judge: checks that the reply names the
    documented reason and offers the documented next step.
    """
    reason = info["reason"].lower()
    next_step = info["next_step"].lower()
    for message in completion:
        if message.role == "assistant" and message.content:
            text = message.content.lower()
            if reason in text and next_step in text:
                return 1.0
    return 0.0


def scripted_agent(*turns: Message):
    """Return an adapter that replays a fixed queue of assistant messages."""
    queue = list(turns)

    def _adapter(messages: list[Message], tools: list[dict]) -> Message:
        return queue.pop(0) if queue else Message(role="assistant", content="")

    return _adapter


# The judge is constructed at import (no key is read until it is called). It is
# shown here to demonstrate the API and used in the live-swap below; it is not
# attached to the offline scenario, so CI never makes a model call.
quality_judge = make_judge(
    "Score 1.0 if the reply politely denies the refund, states the reason, and "
    "offers a concrete next step. Score lower as those elements are missing."
)


scenario = Scenario(
    id="refund_denial_judge",
    system=(
        "You are a support agent. Write a clear, polite reply that denies a "
        "refund on a final-sale item, states the reason, and offers a next step."
    ),
    persona=Persona(goal="Get a refund for a final-sale jacket."),
    opening_message="I want a refund for the jacket I bought on the final-sale rack.",
    max_turns=1,
    info={"reason": "final sale", "next_step": "store credit"},
    rubric=Rubric(funcs=[explains_denial], pass_threshold=0.5),
)

adapter = scripted_agent(
    Message(
        role="assistant",
        content=(
            "I'm sorry, but items marked final sale cannot be refunded. As a "
            "next step, I can offer store credit toward another size or style."
        ),
    ),
)

# Go live with the judge (needs a key): score with the judge instead of, or
# alongside, the deterministic reward, and drive a real agent.
#   from korrel import adapter_from_provider
#   from korrel.providers import AnthropicProvider
#   scenario.rubric = Rubric(funcs=[explains_denial], judge=quality_judge)
#   adapter = adapter_from_provider(AnthropicProvider())
