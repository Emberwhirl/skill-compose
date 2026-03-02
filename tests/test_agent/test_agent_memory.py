"""
Tests for agent-level memory features.

Covers:
- _is_silent_reply() exact matching
- MEMORY_FLUSH_SYSTEM_PROMPT / MEMORY_FLUSH_USER_PROMPT constants
- _build_memory_section() with SOUL.md persona, Memory Recall, citations
- _create_flush_tools() restricted read/write tools scoped to memory directory
- _run_memory_flush_turn() orchestration (NO_REPLY, tool calls, multi-turn)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from typing import Optional

from app.agent.agent import (
    _is_silent_reply,
    MEMORY_FLUSH_SYSTEM_PROMPT,
    MEMORY_FLUSH_USER_PROMPT,
    MEMORY_FLUSH_MAX_TURNS,
)
from app.llm.provider import LLMTextBlock, LLMToolCall, LLMResponse


# ─── _is_silent_reply() ──────────────────────────────────────


class TestIsSilentReply:
    """Tests for _is_silent_reply() — OpenClaw-aligned exact matching."""

    def test_exact_no_reply(self):
        assert _is_silent_reply("NO_REPLY") is True

    def test_leading_whitespace(self):
        assert _is_silent_reply("  NO_REPLY") is True

    def test_leading_newline(self):
        assert _is_silent_reply("\n NO_REPLY") is True

    def test_trailing_whitespace(self):
        """Trailing whitespace should still match."""
        assert _is_silent_reply("NO_REPLY  ") is True

    def test_trailing_text_rejected(self):
        """Exact match: NO_REPLY followed by extra text should NOT match."""
        assert _is_silent_reply("NO_REPLY some extra text") is False

    def test_lowercase_rejected(self):
        """Exact match is case-sensitive."""
        assert _is_silent_reply("no_reply") is False

    def test_mixed_case_rejected(self):
        assert _is_silent_reply("No_Reply") is False

    def test_empty_string(self):
        assert _is_silent_reply("") is False

    def test_whitespace_only(self):
        assert _is_silent_reply("   ") is False

    def test_unrelated_text(self):
        assert _is_silent_reply("I wrote the files") is False

    def test_no_reply_embedded(self):
        """NO_REPLY not at start should not match."""
        assert _is_silent_reply("Result: NO_REPLY") is False


# ─── Prompt Constants ─────────────────────────────────────────


class TestFlushPromptConstants:
    """Verify flush prompts match OpenClaw's expected wording."""

    def test_system_prompt_contains_key_phrases(self):
        assert "Pre-compaction memory flush turn" in MEMORY_FLUSH_SYSTEM_PROMPT
        assert "auto-compaction" in MEMORY_FLUSH_SYSTEM_PROMPT
        assert "NO_REPLY" in MEMORY_FLUSH_SYSTEM_PROMPT

    def test_system_prompt_single_line(self):
        """System prompt should not contain newlines (concise one-liner)."""
        assert "\n" not in MEMORY_FLUSH_SYSTEM_PROMPT

    def test_user_prompt_has_placeholders(self):
        assert "{date}" in MEMORY_FLUSH_USER_PROMPT
        assert "{current_time}" in MEMORY_FLUSH_USER_PROMPT
        assert "{existing_files}" in MEMORY_FLUSH_USER_PROMPT

    def test_user_prompt_append_instruction(self):
        assert "appended" in MEMORY_FLUSH_USER_PROMPT or "APPEND" in MEMORY_FLUSH_USER_PROMPT
        assert "do not overwrite" in MEMORY_FLUSH_USER_PROMPT

    def test_user_prompt_mentions_bootstrap_files(self):
        """Prompt should guide the LLM to write all 3 bootstrap files + daily log."""
        assert "SOUL.md" in MEMORY_FLUSH_USER_PROMPT
        assert "USER.md" in MEMORY_FLUSH_USER_PROMPT
        assert "MEMORY.md" in MEMORY_FLUSH_USER_PROMPT
        assert "memory/{date}.md" in MEMORY_FLUSH_USER_PROMPT

    def test_user_prompt_no_reply_instruction(self):
        assert "NO_REPLY" in MEMORY_FLUSH_USER_PROMPT

    def test_user_prompt_format_succeeds(self):
        """Prompt should format without errors."""
        result = MEMORY_FLUSH_USER_PROMPT.format(
            date="2026-03-02",
            current_time="2026-03-02 14:30:00",
            existing_files="- SOUL.md (42 bytes)",
        )
        assert "2026-03-02" in result
        assert "14:30:00" in result
        assert "SOUL.md" in result

    def test_max_turns_is_3(self):
        assert MEMORY_FLUSH_MAX_TURNS == 3


# ─── _build_memory_section() ─────────────────────────────────


def _make_agent_stub(agent_id="test-agent", verbose=False):
    """Create a minimal agent-like object for testing _build_memory_section."""
    from app.agent.agent import SkillsAgent

    # Bypass __init__ entirely by creating a bare instance
    agent = object.__new__(SkillsAgent)
    agent.agent_id = agent_id
    agent.verbose = verbose
    return agent


class TestBuildMemorySection:
    """Tests for SkillsAgent._build_memory_section()."""

    def test_no_files_returns_empty(self):
        """When no bootstrap files exist, should return empty string."""
        agent = _make_agent_stub()
        with patch("app.services.memory_service.load_bootstrap_files", return_value={}):
            result = agent._build_memory_section()
        assert result == ""

    def test_basic_files_included(self):
        """Bootstrap files should appear as ### sections."""
        agent = _make_agent_stub()
        mock_files = {"MEMORY.md": "some memories", "USER.md": "user prefs"}
        with patch("app.services.memory_service.load_bootstrap_files", return_value=mock_files):
            result = agent._build_memory_section()
        assert "## Agent Memory" in result
        assert "### MEMORY.md" in result
        assert "some memories" in result
        assert "### USER.md" in result
        assert "user prefs" in result

    def test_soul_md_persona_instruction(self):
        """When SOUL.md exists, persona instruction should be included."""
        agent = _make_agent_stub()
        mock_files = {"SOUL.md": "I am creative"}
        with patch("app.services.memory_service.load_bootstrap_files", return_value=mock_files):
            result = agent._build_memory_section()
        assert "embody its persona and tone" in result
        assert "Avoid stiff, generic replies" in result

    def test_no_soul_md_no_persona_instruction(self):
        """When SOUL.md doesn't exist, persona instruction should be absent."""
        agent = _make_agent_stub()
        mock_files = {"MEMORY.md": "facts"}
        with patch("app.services.memory_service.load_bootstrap_files", return_value=mock_files):
            result = agent._build_memory_section()
        assert "embody its persona" not in result

    def test_memory_recall_directive_with_agent_id(self):
        """When agent_id is set, Memory Recall section should be present."""
        agent = _make_agent_stub(agent_id="my-agent")
        mock_files = {"MEMORY.md": "facts"}
        with patch("app.services.memory_service.load_bootstrap_files", return_value=mock_files):
            result = agent._build_memory_section()
        assert "### Memory Recall" in result
        assert "memory_search" in result
        assert "memory_get" in result

    def test_no_memory_recall_without_agent_id(self):
        """When agent_id is falsy, Memory Recall should be absent."""
        agent = _make_agent_stub(agent_id=None)
        mock_files = {"MEMORY.md": "facts"}
        with patch("app.services.memory_service.load_bootstrap_files", return_value=mock_files):
            result = agent._build_memory_section()
        assert "### Memory Recall" not in result

    def test_citations_guidance_present(self):
        """Citations instruction should be present in Memory Recall."""
        agent = _make_agent_stub(agent_id="my-agent")
        mock_files = {"MEMORY.md": "facts"}
        with patch("app.services.memory_service.load_bootstrap_files", return_value=mock_files):
            result = agent._build_memory_section()
        assert "Source: <path#line>" in result

    def test_persona_before_file_contents(self):
        """SOUL.md persona instruction should appear before file contents."""
        agent = _make_agent_stub()
        mock_files = {"SOUL.md": "CONTENT_HERE"}
        with patch("app.services.memory_service.load_bootstrap_files", return_value=mock_files):
            result = agent._build_memory_section()
        persona_pos = result.index("embody its persona")
        content_pos = result.index("CONTENT_HERE")
        assert persona_pos < content_pos

    def test_exception_returns_empty(self):
        """If loading fails, should return empty string."""
        agent = _make_agent_stub(verbose=True)
        with patch("app.services.memory_service.load_bootstrap_files", side_effect=RuntimeError("boom")):
            result = agent._build_memory_section()
        assert result == ""


# ─── _create_flush_tools() ────────────────────────────────────


class TestCreateFlushTools:
    """Tests for SkillsAgent._create_flush_tools()."""

    def _make_agent_with_tools(self, tmp_path, agent_id="test-agent"):
        """Create agent stub with self.tools and self.tool_functions set."""
        from app.agent.agent import SkillsAgent
        agent = object.__new__(SkillsAgent)
        agent.agent_id = agent_id
        agent.verbose = False

        # Simulate agent's own tools
        agent.tools = [
            {"name": "read", "description": "original read"},
            {"name": "write", "description": "original write"},
            {"name": "execute_code", "description": "run code"},
        ]
        agent.tool_functions = {
            "read": lambda **kw: "original_read",
            "write": lambda **kw: "original_write",
            "execute_code": lambda **kw: "executed",
        }
        return agent

    def test_returns_only_read_write(self, tmp_path):
        """Flush tools should only include read + write with dedicated schemas."""
        agent = self._make_agent_with_tools(tmp_path)
        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            tools, funcs = agent._create_flush_tools()
        tool_names = [t["name"] for t in tools]
        assert tool_names == ["read", "write"]
        assert "execute_code" not in tool_names

    def test_dedicated_schemas_describe_relative_paths(self, tmp_path):
        """Flush tool schemas should describe relative paths, not absolute."""
        agent = self._make_agent_with_tools(tmp_path)
        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            tools, funcs = agent._create_flush_tools()
        read_schema = tools[0]
        assert "relative" in read_schema["description"].lower() or "memory" in read_schema["description"].lower()
        assert "SOUL.md" in str(read_schema["input_schema"])

    def test_other_tools_excluded(self, tmp_path):
        """Non-read/write tools should NOT be in flush tools."""
        agent = self._make_agent_with_tools(tmp_path)
        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            tools, funcs = agent._create_flush_tools()
        assert "execute_code" not in funcs
        assert len(tools) == 2

    def test_flush_read_reads_file(self, tmp_path):
        """Scoped read should read from memory directory and return JSON."""
        agent = self._make_agent_with_tools(tmp_path)
        agent_dir = tmp_path / "agents" / "test-agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "SOUL.md").write_text("soul data")

        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            _, funcs = agent._create_flush_tools()
        result = funcs["read"](file_path="SOUL.md")
        parsed = json.loads(result)
        assert parsed["content"] == "soul data"

    def test_flush_read_nonexistent(self, tmp_path):
        """Scoped read of nonexistent file should return error JSON."""
        agent = self._make_agent_with_tools(tmp_path)
        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            _, funcs = agent._create_flush_tools()
        result = funcs["read"](file_path="NOPE.md")
        assert "error" in json.loads(result)

    def test_flush_write_creates_file(self, tmp_path):
        """Scoped write should create file in memory directory."""
        agent = self._make_agent_with_tools(tmp_path)
        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            _, funcs = agent._create_flush_tools()
        result = funcs["write"](file_path="memory/2026-03-02.md", content="notes")
        parsed = json.loads(result)
        assert parsed["success"] is True

        written = (tmp_path / "agents" / "test-agent" / "memory" / "2026-03-02.md").read_text()
        assert written == "notes"

    def test_flush_read_path_traversal_blocked(self, tmp_path):
        """Path traversal in read should be denied."""
        agent = self._make_agent_with_tools(tmp_path)
        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            _, funcs = agent._create_flush_tools()
        result = funcs["read"](file_path="../../etc/passwd")
        assert "Access denied" in result

    def test_flush_write_path_traversal_blocked(self, tmp_path):
        """Path traversal in write should be denied."""
        agent = self._make_agent_with_tools(tmp_path)
        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            _, funcs = agent._create_flush_tools()
        result = funcs["write"](file_path="../../etc/evil", content="bad")
        assert "Access denied" in result


# ─── _run_memory_flush_turn() ─────────────────────────────────


def _make_no_reply_response():
    """Create an LLMResponse with NO_REPLY text."""
    return LLMResponse(content=[LLMTextBlock(text="NO_REPLY")])


def _make_text_response(text):
    """Create an LLMResponse with text content."""
    return LLMResponse(content=[LLMTextBlock(text=text)])


def _make_tool_response(tool_id, name, input_data):
    """Create an LLMResponse with a tool call."""
    return LLMResponse(content=[LLMToolCall(id=tool_id, name=name, input=input_data)])


def _make_flush_agent(tmp_path, agent_id="test-agent"):
    """Create an agent stub suitable for _run_memory_flush_turn tests."""
    from app.agent.agent import SkillsAgent

    agent = object.__new__(SkillsAgent)
    agent.agent_id = agent_id
    agent.verbose = False
    agent.system_prompt = "You are a helpful assistant."

    # Minimal tool set
    agent.tools = [
        {"name": "read", "description": "read"},
        {"name": "write", "description": "write"},
    ]
    agent.tool_functions = {
        "read": lambda **kw: "original",
        "write": lambda **kw: "original",
    }

    # Mock LLM client
    agent.client = MagicMock()
    agent.client.acreate = AsyncMock()

    return agent


class TestRunMemoryFlushTurn:
    """Tests for SkillsAgent._run_memory_flush_turn()."""

    @pytest.mark.asyncio
    async def test_no_reply_exits_immediately(self, tmp_path):
        """When LLM responds with NO_REPLY, flush should exit without tool calls."""
        agent = _make_flush_agent(tmp_path)
        agent.client.acreate.return_value = _make_no_reply_response()

        messages = [{"role": "user", "content": "hello"}]
        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            await agent._run_memory_flush_turn(messages)

        assert agent.client.acreate.call_count == 1

    @pytest.mark.asyncio
    async def test_no_reply_with_whitespace(self, tmp_path):
        """NO_REPLY with surrounding whitespace should be treated as silent."""
        agent = _make_flush_agent(tmp_path)
        agent.client.acreate.return_value = _make_text_response("  NO_REPLY  ")

        messages = [{"role": "user", "content": "hello"}]
        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            await agent._run_memory_flush_turn(messages)

        assert agent.client.acreate.call_count == 1

    @pytest.mark.asyncio
    async def test_text_response_not_no_reply_exits(self, tmp_path):
        """Text response that's not NO_REPLY should still exit (agent done talking)."""
        agent = _make_flush_agent(tmp_path)
        agent.client.acreate.return_value = _make_text_response("I saved the memories.")

        messages = [{"role": "user", "content": "hello"}]
        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            await agent._run_memory_flush_turn(messages)

        assert agent.client.acreate.call_count == 1

    @pytest.mark.asyncio
    async def test_full_conversation_passed(self, tmp_path):
        """Flush should pass full conversation messages, not serialized text."""
        agent = _make_flush_agent(tmp_path)
        agent.client.acreate.return_value = _make_no_reply_response()

        messages = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "first reply"},
            {"role": "user", "content": "second message"},
        ]
        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            await agent._run_memory_flush_turn(messages)

        call_kwargs = agent.client.acreate.call_args
        flush_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        # Should contain original messages + flush user message = 4 total
        assert len(flush_messages) == 4
        assert flush_messages[0]["content"] == "first message"
        assert flush_messages[1]["content"] == "first reply"
        assert flush_messages[2]["content"] == "second message"
        # Last message is the flush prompt
        assert "Pre-compaction memory flush" in flush_messages[3]["content"]

    @pytest.mark.asyncio
    async def test_system_prompt_includes_agent_system(self, tmp_path):
        """Flush system prompt should include agent's full system prompt."""
        agent = _make_flush_agent(tmp_path)
        agent.system_prompt = "AGENT_SYSTEM_PROMPT_HERE"
        agent.client.acreate.return_value = _make_no_reply_response()

        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            await agent._run_memory_flush_turn([{"role": "user", "content": "hi"}])

        call_kwargs = agent.client.acreate.call_args
        system = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        assert "AGENT_SYSTEM_PROMPT_HERE" in system
        assert "Pre-compaction memory flush turn" in system

    @pytest.mark.asyncio
    async def test_tool_call_executed(self, tmp_path):
        """When LLM makes tool calls, they should be executed."""
        agent = _make_flush_agent(tmp_path)

        # Ensure memory directory exists for write
        agent_dir = tmp_path / "agents" / "test-agent" / "memory"
        agent_dir.mkdir(parents=True)

        # First response: tool call to write
        resp1 = _make_tool_response(
            "tc_1", "write",
            {"file_path": "memory/2026-03-02.md", "content": "test notes"},
        )
        # Second response: NO_REPLY (done)
        resp2 = _make_no_reply_response()
        agent.client.acreate.side_effect = [resp1, resp2]

        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            await agent._run_memory_flush_turn([{"role": "user", "content": "hi"}])

        # File should have been written
        written_path = tmp_path / "agents" / "test-agent" / "memory" / "2026-03-02.md"
        assert written_path.exists()
        assert written_path.read_text() == "test notes"

        # Two LLM calls: first with tool call, second with NO_REPLY
        assert agent.client.acreate.call_count == 2

    @pytest.mark.asyncio
    async def test_multi_turn_tool_calls(self, tmp_path):
        """Multiple rounds of tool calls should work."""
        agent = _make_flush_agent(tmp_path)
        agent_dir = tmp_path / "agents" / "test-agent"
        agent_dir.mkdir(parents=True)

        # Turn 1: write SOUL.md
        resp1 = _make_tool_response("tc_1", "write", {"file_path": "SOUL.md", "content": "I am creative"})
        # Turn 2: read it back
        resp2 = _make_tool_response("tc_2", "read", {"file_path": "SOUL.md"})
        # Turn 3: done
        resp3 = _make_no_reply_response()
        agent.client.acreate.side_effect = [resp1, resp2, resp3]

        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            await agent._run_memory_flush_turn([{"role": "user", "content": "hi"}])

        assert agent.client.acreate.call_count == 3

    @pytest.mark.asyncio
    async def test_max_turns_limit(self, tmp_path):
        """Flush should stop after MEMORY_FLUSH_MAX_TURNS even if LLM keeps calling tools."""
        agent = _make_flush_agent(tmp_path)
        agent_dir = tmp_path / "agents" / "test-agent" / "memory"
        agent_dir.mkdir(parents=True)

        # Every turn returns a tool call — never NO_REPLY
        agent.client.acreate.side_effect = [
            _make_tool_response(
                f"tc_{i}", "write",
                {"file_path": f"memory/note-{i}.md", "content": f"note {i}"},
            )
            for i in range(MEMORY_FLUSH_MAX_TURNS + 5)
        ]

        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            await agent._run_memory_flush_turn([{"role": "user", "content": "hi"}])

        assert agent.client.acreate.call_count == MEMORY_FLUSH_MAX_TURNS

    @pytest.mark.asyncio
    async def test_flush_user_prompt_contains_current_time(self, tmp_path):
        """Flush user message should contain formatted current time."""
        agent = _make_flush_agent(tmp_path)
        agent.client.acreate.return_value = _make_no_reply_response()

        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            await agent._run_memory_flush_turn([{"role": "user", "content": "hi"}])

        call_kwargs = agent.client.acreate.call_args
        flush_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        last_msg = flush_messages[-1]["content"]
        assert "Current time:" in last_msg

    @pytest.mark.asyncio
    async def test_original_messages_not_mutated(self, tmp_path):
        """Original messages list should not be modified."""
        agent = _make_flush_agent(tmp_path)
        agent.client.acreate.return_value = _make_no_reply_response()

        messages = [{"role": "user", "content": "hello"}]
        original_len = len(messages)

        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            await agent._run_memory_flush_turn(messages)

        assert len(messages) == original_len

    @pytest.mark.asyncio
    async def test_tool_results_appended_to_flush_messages(self, tmp_path):
        """After tool execution, results should be appended for next LLM call."""
        agent = _make_flush_agent(tmp_path)
        agent_dir = tmp_path / "agents" / "test-agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "SOUL.md").write_text("existing soul")

        # Turn 1: read SOUL.md
        resp1 = _make_tool_response("tc_1", "read", {"file_path": "SOUL.md"})
        # Turn 2: done
        resp2 = _make_no_reply_response()
        agent.client.acreate.side_effect = [resp1, resp2]

        with patch("app.services.memory_service._memory_dir", return_value=tmp_path):
            await agent._run_memory_flush_turn([{"role": "user", "content": "hi"}])

        # Second call should have tool result in messages
        second_call_kwargs = agent.client.acreate.call_args_list[1]
        msgs = second_call_kwargs.kwargs.get("messages") or second_call_kwargs[1].get("messages")

        # Should contain: original user msg, flush prompt, assistant tool_use, user tool_result
        assert len(msgs) >= 4
        # Last user message should contain tool_result
        last_user = msgs[-1]
        assert last_user["role"] == "user"
        tool_results = last_user["content"]
        assert isinstance(tool_results, list)
        assert tool_results[0]["type"] == "tool_result"
        assert "existing soul" in tool_results[0]["content"]
