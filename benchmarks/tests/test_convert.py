"""Tests for benchmarks/fidelity/_convert.py.

Covers:
- tau2 -> korrel direction for all five message types
- korrel -> tau2 reverse direction
- arguments encoding: dict <-> JSON-string round-trip
- MultiToolMessage flattening
- None filtering in tau2_messages_to_korrel
- Audio message placeholder
- Malformed-JSON fallback in korrel_to_tau2_tool_call
- Property-based round-trip with hypothesis
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from fidelity._convert import (
    korrel_message_to_tau2,
    korrel_messages_to_tau2,
    korrel_to_tau2_tool_call,
    tau2_message_to_korrel,
    tau2_messages_to_korrel,
    tau2_tool_call_to_korrel,
)
from korrel.types import Message, ToolCall as KorrelToolCall, ToolFunction

from tests.fixtures.tau2_builders import (
    make_multi_tool_message,
    make_tool_call,
    make_tool_message,
    make_minimal_message_sequence,
)


# ---------------------------------------------------------------------------
# Arguments encoding
# ---------------------------------------------------------------------------


class TestArgumentsEncoding:
    """dict <-> JSON-string round-trip for the tool-call arguments field."""

    @pytest.mark.parametrize(
        "arguments",
        [
            {},
            {"key": "value"},
            {"nested": {"a": 1, "b": [2, 3]}},
            {"unicode": "中文内容"},
            {"empty_string": ""},
            {"null_value": None},
            {"bool_val": True, "int_val": 42, "float_val": 3.14},
            {"list_val": [1, "two", {"three": 3}]},
        ],
        ids=[
            "empty",
            "flat",
            "nested",
            "unicode",
            "empty_string",
            "null_value",
            "primitives",
            "nested_list",
        ],
    )
    def test_tau2_to_korrel_arguments_is_json_string(self, arguments: dict[str, Any]) -> None:
        tc = make_tool_call("tc1", "fn", arguments)
        korrel_tc = tau2_tool_call_to_korrel(tc)
        # arguments must be a string, not a dict
        assert isinstance(korrel_tc.function.arguments, str)
        # must be valid JSON that round-trips back to the original dict
        decoded = json.loads(korrel_tc.function.arguments)
        assert decoded == arguments

    @pytest.mark.parametrize(
        "arguments",
        [
            {},
            {"key": "value"},
            {"nested": {"x": [1, 2, 3]}},
            {"unicode": "élève"},
        ],
        ids=["empty", "flat", "nested", "unicode"],
    )
    def test_korrel_to_tau2_arguments_is_dict(self, arguments: dict[str, Any]) -> None:
        json_str = json.dumps(arguments, ensure_ascii=False)
        korrel_tc = KorrelToolCall(
            id="tc1",
            type="function",
            function=ToolFunction(name="fn", arguments=json_str),
        )
        tau2_tc = korrel_to_tau2_tool_call(korrel_tc)
        assert isinstance(tau2_tc.arguments, dict)
        assert tau2_tc.arguments == arguments

    def test_malformed_json_falls_back_to_raw_dict(self) -> None:
        """korrel_to_tau2_tool_call gracefully handles non-JSON arguments strings."""
        korrel_tc = KorrelToolCall(
            id="tc1",
            type="function",
            function=ToolFunction(name="fn", arguments="NOT VALID JSON {{{"),
        )
        tau2_tc = korrel_to_tau2_tool_call(korrel_tc)
        assert isinstance(tau2_tc.arguments, dict)
        assert "_raw" in tau2_tc.arguments
        assert tau2_tc.arguments["_raw"] == "NOT VALID JSON {{{"

    def test_ensure_ascii_false_preserves_unicode_in_arguments(self) -> None:
        """tau2 dict with unicode values -> JSON string that contains unicode directly."""
        arguments = {"greeting": "你好"}
        tc = make_tool_call("tc1", "say", arguments)
        korrel_tc = tau2_tool_call_to_korrel(tc)
        # The JSON string must NOT escape the unicode (ensure_ascii=False)
        assert "你好" in korrel_tc.function.arguments


# ---------------------------------------------------------------------------
# tau2 -> korrel per message type
# ---------------------------------------------------------------------------


class TestTau2ToKorrel:
    """Conversion of individual tau2 message types to korrel Messages."""

    def test_system_message(self) -> None:
        msg = SystemMessage(role="system", content="You are helpful.")
        result = tau2_message_to_korrel(msg)
        assert len(result) == 1
        m = result[0]
        assert m.role == "system"
        assert m.content == "You are helpful."
        assert m.tool_calls is None
        assert m.tool_call_id is None

    def test_user_message_no_tool_calls(self) -> None:
        msg = UserMessage(role="user", content="Hello")
        result = tau2_message_to_korrel(msg)
        assert len(result) == 1
        m = result[0]
        assert m.role == "user"
        assert m.content == "Hello"
        assert m.tool_calls is None

    def test_assistant_message_no_tool_calls(self) -> None:
        msg = AssistantMessage(role="assistant", content="Hi there")
        result = tau2_message_to_korrel(msg)
        assert len(result) == 1
        m = result[0]
        assert m.role == "assistant"
        assert m.content == "Hi there"
        assert m.tool_calls is None

    def test_assistant_message_with_tool_calls(self) -> None:
        tc = make_tool_call("tc1", "search", {"q": "order 123"})
        msg = AssistantMessage(role="assistant", content=None, tool_calls=[tc])
        result = tau2_message_to_korrel(msg)
        assert len(result) == 1
        m = result[0]
        assert m.role == "assistant"
        assert m.content is None
        assert m.tool_calls is not None
        assert len(m.tool_calls) == 1
        assert m.tool_calls[0].id == "tc1"
        assert m.tool_calls[0].function.name == "search"
        # arguments must be a JSON string
        assert isinstance(m.tool_calls[0].function.arguments, str)
        assert json.loads(m.tool_calls[0].function.arguments) == {"q": "order 123"}

    def test_user_message_with_tool_calls(self) -> None:
        """UserMessage can also carry tool_calls (tau2 half-duplex allows it)."""
        tc = make_tool_call("tc-u", "user_tool", {"x": 1}, requestor="user")
        msg = UserMessage(role="user", content=None, tool_calls=[tc])
        result = tau2_message_to_korrel(msg)
        assert len(result) == 1
        m = result[0]
        assert m.role == "user"
        assert m.tool_calls is not None
        assert m.tool_calls[0].id == "tc-u"

    def test_tool_message_id_mapped_to_tool_call_id(self) -> None:
        """tau2 ToolMessage.id must land on korrel Message.tool_call_id."""
        msg = make_tool_message("tc-42", "response text")
        result = tau2_message_to_korrel(msg)
        assert len(result) == 1
        m = result[0]
        assert m.role == "tool"
        assert m.tool_call_id == "tc-42"
        assert m.content == "response text"
        # tau2 ToolMessage has no .name; korrel name must be None
        assert m.name is None

    def test_tool_message_null_content(self) -> None:
        msg = ToolMessage(id="tc1", role="tool", content=None, requestor="assistant")
        result = tau2_message_to_korrel(msg)
        assert result[0].content is None

    def test_multi_tool_message_flattens_to_individual_messages(self) -> None:
        mtm = make_multi_tool_message(
            [
                make_tool_message("tc1", "r1"),
                make_tool_message("tc2", "r2"),
                make_tool_message("tc3", "r3"),
            ]
        )
        result = tau2_message_to_korrel(mtm)
        assert len(result) == 3
        assert [m.tool_call_id for m in result] == ["tc1", "tc2", "tc3"]
        assert [m.content for m in result] == ["r1", "r2", "r3"]
        for m in result:
            assert m.role == "tool"

    def test_multi_tool_message_single_inner(self) -> None:
        mtm = make_multi_tool_message([make_tool_message("only", "solo result")])
        result = tau2_message_to_korrel(mtm)
        assert len(result) == 1
        assert result[0].tool_call_id == "only"

    def test_audio_message_with_no_content_gets_placeholder(self) -> None:
        msg = AssistantMessage(role="assistant", is_audio=True, content=None)
        result = tau2_message_to_korrel(msg)
        assert result[0].content == "[audio]"

    def test_audio_message_with_content_keeps_content(self) -> None:
        msg = AssistantMessage(role="assistant", is_audio=True, content="Transcript text")
        result = tau2_message_to_korrel(msg)
        assert result[0].content == "Transcript text"

    def test_tau2_messages_to_korrel_filters_none(self) -> None:
        msg = UserMessage(role="user", content="Hello")
        result = tau2_messages_to_korrel([None, msg, None])
        assert len(result) == 1
        assert result[0].content == "Hello"

    def test_tau2_messages_to_korrel_empty_list(self) -> None:
        assert tau2_messages_to_korrel([]) == []

    def test_tau2_messages_to_korrel_all_none(self) -> None:
        assert tau2_messages_to_korrel([None, None]) == []

    def test_multi_tool_message_flattened_in_sequence(self) -> None:
        """MultiToolMessage in a sequence contributes N messages, not 1."""
        mtm = make_multi_tool_message([
            make_tool_message("tc1", "r1"),
            make_tool_message("tc2", "r2"),
        ])
        msgs = [
            UserMessage(role="user", content="Hi"),
            mtm,
        ]
        result = tau2_messages_to_korrel(msgs)
        assert len(result) == 3
        assert result[0].role == "user"
        assert result[1].role == "tool"
        assert result[2].role == "tool"


# ---------------------------------------------------------------------------
# korrel -> tau2 per message type
# ---------------------------------------------------------------------------


class TestKorrelToTau2:
    """Conversion of korrel Messages back to tau2 message objects."""

    def test_system_message(self) -> None:
        msg = Message(role="system", content="System prompt")
        result = korrel_message_to_tau2(msg)
        assert isinstance(result, SystemMessage)
        assert result.role == "system"
        assert result.content == "System prompt"

    def test_user_message_no_tool_calls(self) -> None:
        msg = Message(role="user", content="Hello")
        result = korrel_message_to_tau2(msg)
        assert isinstance(result, UserMessage)
        assert result.content == "Hello"
        assert result.tool_calls is None

    def test_assistant_message_no_tool_calls(self) -> None:
        msg = Message(role="assistant", content="Hi there")
        result = korrel_message_to_tau2(msg)
        assert isinstance(result, AssistantMessage)
        assert result.content == "Hi there"
        assert result.tool_calls is None

    def test_tool_message_tool_call_id_maps_to_id(self) -> None:
        """korrel Message.tool_call_id must land on tau2 ToolMessage.id."""
        msg = Message(role="tool", content="result", tool_call_id="tc-99")
        result = korrel_message_to_tau2(msg)
        assert isinstance(result, ToolMessage)
        assert result.id == "tc-99"
        assert result.content == "result"

    def test_tool_message_none_tool_call_id_becomes_empty_string(self) -> None:
        msg = Message(role="tool", content="result", tool_call_id=None)
        result = korrel_message_to_tau2(msg)
        assert isinstance(result, ToolMessage)
        assert result.id == ""

    def test_assistant_message_with_tool_calls(self) -> None:
        korrel_tc = KorrelToolCall(
            id="tc1",
            type="function",
            function=ToolFunction(name="fn", arguments='{"k": "v"}'),
        )
        msg = Message(role="assistant", content=None, tool_calls=[korrel_tc])
        result = korrel_message_to_tau2(msg)
        assert isinstance(result, AssistantMessage)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tc1"
        assert result.tool_calls[0].name == "fn"
        assert result.tool_calls[0].arguments == {"k": "v"}

    def test_korrel_messages_to_tau2_empty(self) -> None:
        assert korrel_messages_to_tau2([]) == []


# ---------------------------------------------------------------------------
# Round-trip: tau2 -> korrel -> tau2
# ---------------------------------------------------------------------------


class TestRoundTripTau2ToKorrelToTau2:
    """tau2 -> korrel -> tau2 preserves fields the evaluator reads."""

    def test_system_message_round_trip(self) -> None:
        original = SystemMessage(role="system", content="System prompt")
        korrel = tau2_message_to_korrel(original)
        restored = [korrel_message_to_tau2(m) for m in korrel]
        assert len(restored) == 1
        r = restored[0]
        assert isinstance(r, SystemMessage)
        assert r.content == original.content

    def test_user_message_round_trip(self) -> None:
        original = UserMessage(role="user", content="User turn text")
        korrel = tau2_message_to_korrel(original)
        restored = korrel_message_to_tau2(korrel[0])
        assert isinstance(restored, UserMessage)
        assert restored.content == original.content

    def test_assistant_message_round_trip(self) -> None:
        tc = make_tool_call("tc1", "search", {"q": "hello"})
        original = AssistantMessage(role="assistant", content=None, tool_calls=[tc])
        korrel = tau2_message_to_korrel(original)
        restored = korrel_message_to_tau2(korrel[0])
        assert isinstance(restored, AssistantMessage)
        assert restored.content == original.content
        assert restored.tool_calls is not None
        assert len(restored.tool_calls) == 1
        rtc = restored.tool_calls[0]
        assert rtc.id == tc.id
        assert rtc.name == tc.name
        assert rtc.arguments == tc.arguments

    def test_tool_message_round_trip(self) -> None:
        original = make_tool_message("tc-42", "tool response")
        korrel = tau2_message_to_korrel(original)
        restored = korrel_message_to_tau2(korrel[0])
        assert isinstance(restored, ToolMessage)
        # evaluator reads: id, content
        assert restored.id == original.id
        assert restored.content == original.content

    def test_multi_tool_message_round_trip_preserves_tool_ids(self) -> None:
        """MultiToolMessage flattens to N korrel messages; round-trip preserves ids."""
        tm1 = make_tool_message("tc1", "r1")
        tm2 = make_tool_message("tc2", "r2")
        mtm = make_multi_tool_message([tm1, tm2])
        korrel_msgs = tau2_message_to_korrel(mtm)
        # Two korrel messages back
        restored = [korrel_message_to_tau2(m) for m in korrel_msgs]
        assert len(restored) == 2
        assert isinstance(restored[0], ToolMessage)
        assert isinstance(restored[1], ToolMessage)
        assert restored[0].id == "tc1"
        assert restored[1].id == "tc2"
        assert restored[0].content == "r1"
        assert restored[1].content == "r2"

    def test_full_sequence_round_trip(self) -> None:
        """Full half-duplex sequence: every message identity is preserved."""
        original_sequence = make_minimal_message_sequence()
        korrel_msgs = tau2_messages_to_korrel(original_sequence)
        restored = korrel_messages_to_tau2(korrel_msgs)
        assert len(restored) == len(original_sequence)
        # roles must match
        for orig, rest in zip(original_sequence, restored):
            assert orig.role == rest.role


# ---------------------------------------------------------------------------
# Round-trip: korrel -> tau2 -> korrel
# ---------------------------------------------------------------------------


class TestRoundTripKorrelToTau2ToKorrel:
    """korrel -> tau2 -> korrel is stable (no data loss in the fields korrel exposes)."""

    @pytest.mark.parametrize(
        "msg",
        [
            Message(role="system", content="s"),
            Message(role="user", content="u"),
            Message(role="assistant", content="a"),
            Message(role="assistant", content=None, tool_calls=[
                KorrelToolCall(
                    id="x",
                    type="function",
                    function=ToolFunction(name="f", arguments='{"a": 1}'),
                )
            ]),
            Message(role="tool", content="result", tool_call_id="tc1"),
        ],
        ids=["system", "user", "assistant_text", "assistant_tool_call", "tool"],
    )
    def test_korrel_to_tau2_to_korrel_stable(self, msg: Message) -> None:
        tau2_msg = korrel_message_to_tau2(msg)
        korrel_restored = tau2_message_to_korrel(tau2_msg)
        assert len(korrel_restored) == 1
        r = korrel_restored[0]
        assert r.role == msg.role
        assert r.content == msg.content
        # For tool messages, tool_call_id must survive
        if msg.role == "tool":
            assert r.tool_call_id == msg.tool_call_id
        # For assistant with tool_calls, arguments must survive as JSON string
        if msg.tool_calls:
            assert r.tool_calls is not None
            for orig_tc, rest_tc in zip(msg.tool_calls, r.tool_calls):
                assert rest_tc.id == orig_tc.id
                assert rest_tc.function.name == orig_tc.function.name
                assert json.loads(rest_tc.function.arguments) == json.loads(
                    orig_tc.function.arguments
                )


# ---------------------------------------------------------------------------
# Property-based round-trips (hypothesis)
# ---------------------------------------------------------------------------

# Strategies for generating tau2-shaped arguments dicts
_json_primitives = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=20),
    st.booleans(),
    st.none(),
)

_json_dict = st.dictionaries(
    keys=st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), min_size=1, max_size=10),
    values=_json_primitives,
    max_size=6,
)


@given(arguments=_json_dict)
@settings(max_examples=150)
def test_property_tool_call_arguments_round_trip(arguments: dict[str, Any]) -> None:
    """tau2 ToolCall.arguments (dict) -> korrel (JSON str) -> tau2 (dict) is lossless."""
    tc = make_tool_call("tc1", "fn", arguments)
    korrel_tc = tau2_tool_call_to_korrel(tc)
    assert isinstance(korrel_tc.function.arguments, str)
    recovered = json.loads(korrel_tc.function.arguments)
    assert recovered == arguments

    tau2_tc_back = korrel_to_tau2_tool_call(korrel_tc, requestor="assistant")
    assert tau2_tc_back.arguments == arguments


@given(
    content=st.one_of(st.none(), st.text(max_size=100)),
    tc_id=st.text(min_size=1, max_size=20),
    tool_name=st.text(min_size=1, max_size=30),
    arguments=_json_dict,
)
@settings(max_examples=120)
def test_property_assistant_message_with_tool_call_round_trip(
    content: str | None,
    tc_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    """tau2 AssistantMessage with a single tool call round-trips through korrel."""
    tc = make_tool_call(tc_id, tool_name, arguments)
    original = AssistantMessage(role="assistant", content=content, tool_calls=[tc])
    korrel_msgs = tau2_message_to_korrel(original)
    assert len(korrel_msgs) == 1
    restored = korrel_message_to_tau2(korrel_msgs[0])
    assert isinstance(restored, AssistantMessage)
    assert restored.content == content
    assert restored.tool_calls is not None
    rtc = restored.tool_calls[0]
    assert rtc.id == tc_id
    assert rtc.name == tool_name
    assert rtc.arguments == arguments


@given(
    tc_id=st.text(min_size=1, max_size=20),
    content=st.one_of(st.none(), st.text(max_size=100)),
)
@settings(max_examples=120)
def test_property_tool_message_id_preserved(tc_id: str, content: str | None) -> None:
    """tau2 ToolMessage.id survives the tau2->korrel->tau2 trip as tool_call_id."""
    original = ToolMessage(id=tc_id, role="tool", content=content, requestor="assistant")
    korrel_msgs = tau2_message_to_korrel(original)
    assert len(korrel_msgs) == 1
    assert korrel_msgs[0].tool_call_id == tc_id
    restored = korrel_message_to_tau2(korrel_msgs[0])
    assert isinstance(restored, ToolMessage)
    assert restored.id == tc_id


@given(
    tool_messages=st.lists(
        st.builds(
            lambda tid, cont: make_tool_message(tid, cont),
            tid=st.text(min_size=1, max_size=15),
            cont=st.text(max_size=60),
        ),
        min_size=1,
        max_size=6,
    )
)
@settings(max_examples=80)
def test_property_multi_tool_message_flattens_correctly(
    tool_messages: list[ToolMessage],
) -> None:
    """MultiToolMessage with N inner messages flattens to exactly N korrel Messages."""
    mtm = make_multi_tool_message(tool_messages)
    korrel_msgs = tau2_message_to_korrel(mtm)
    assert len(korrel_msgs) == len(tool_messages)
    for km, tm in zip(korrel_msgs, tool_messages):
        assert km.role == "tool"
        assert km.tool_call_id == tm.id


@given(
    sequence=st.lists(
        st.one_of(
            st.builds(
                lambda c: SystemMessage(role="system", content=c),
                c=st.text(max_size=50),
            ),
            st.builds(
                lambda c: UserMessage(role="user", content=c),
                c=st.text(max_size=50),
            ),
            st.builds(
                lambda c: AssistantMessage(role="assistant", content=c),
                c=st.text(max_size=50),
            ),
        ),
        min_size=0,
        max_size=8,
    )
)
@settings(max_examples=80)
def test_property_sequence_role_order_preserved(sequence: list[Any]) -> None:
    """tau2_messages_to_korrel followed by korrel_messages_to_tau2 preserves roles."""
    korrel_msgs = tau2_messages_to_korrel(sequence)
    restored = korrel_messages_to_tau2(korrel_msgs)
    assert len(restored) == len(sequence)
    for orig, rest in zip(sequence, restored):
        assert orig.role == rest.role
