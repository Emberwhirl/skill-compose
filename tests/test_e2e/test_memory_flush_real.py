"""
Real E2E test for memory flush turn.

Runs an actual LLM call (Kimi K2.5) to verify the flush turn writes
to memory files (SOUL.md, USER.md, MEMORY.md, memory/*.md).

Usage:
    MOONSHOT_API_KEY=sk-xxx python -m pytest tests/test_e2e/test_memory_flush_real.py -v -s
"""

import json
import os
import uuid
import pytest
from pathlib import Path

# Skip entire module if no API key
MOONSHOT_KEY = os.environ.get("MOONSHOT_API_KEY", "")
pytestmark = pytest.mark.skipif(
    not MOONSHOT_KEY,
    reason="MOONSHOT_API_KEY not set — skipping real LLM test",
)


@pytest.fixture
def memory_root(tmp_path):
    """Temporary memory root directory."""
    return tmp_path


@pytest.fixture
def agent_id():
    return str(uuid.uuid4())


def _build_rich_conversation():
    """Build a conversation with clear signals for all 3 memory files."""
    return [
        {
            "role": "user",
            "content": (
                "Hi! I'm Alex. I'm a backend developer who loves Python and FastAPI. "
                "I always prefer dark mode, concise answers, and code over prose. "
                "Please remember these preferences for our future sessions."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Got it, Alex! I'll keep things concise and code-focused. "
                "Dark mode + Python + FastAPI — noted for all future conversations."
            ),
        },
        {
            "role": "user",
            "content": (
                "I'd also like you to adopt a specific persona: be a senior staff engineer. "
                "Use precise technical language, avoid filler words, and be direct. "
                "When reviewing code, be constructively critical."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Understood. I'll operate as a senior staff engineer — "
                "direct, precise, constructively critical. No fluff."
            ),
        },
        {
            "role": "user",
            "content": (
                "Today we made an important architectural decision: we're migrating from "
                "REST to gRPC for inter-service communication. The migration starts on "
                "March 10, 2026. We're using protobuf v3 and grpcio 1.60. "
                "We also decided to keep REST for external-facing APIs only. "
                "Can you help me draft the proto files tomorrow?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Good call — gRPC for internal, REST for external is a solid pattern. "
                "Noted: migration starts March 10, protobuf v3, grpcio 1.60. "
                "Tomorrow I'll help with proto file structure. "
                "I'd suggest starting with the service definitions that have "
                "the highest call volume."
            ),
        },
    ]


@pytest.mark.asyncio
async def test_memory_flush_writes_files(memory_root, agent_id):
    """Run a real flush turn and verify that memory files are created.

    The conversation contains:
    - User preferences (name, dark mode, Python) → USER.md
    - Persona instructions (senior staff engineer) → SOUL.md
    - Architectural facts (gRPC migration, dates) → MEMORY.md / memory/*.md
    """
    from unittest.mock import patch

    from app.agent.agent import SkillsAgent
    from app.llm import LLMClient

    # Create a minimal agent with real LLM client
    agent = object.__new__(SkillsAgent)
    agent.agent_id = agent_id
    agent.verbose = True
    agent.system_prompt = (
        "You are a helpful assistant. When asked to store memories, "
        "write to the appropriate files: SOUL.md for persona/tone, "
        "USER.md for user preferences, MEMORY.md for important facts."
    )

    # Real LLM client
    agent.client = LLMClient(provider="kimi", model="kimi-k2.5")

    # Build tool set: only read + write (scoped to memory dir)
    agent.tools = [
        {
            "name": "read",
            "description": "Read a memory file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path within memory directory (e.g. 'SOUL.md', 'memory/2026-03-02.md')",
                    },
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "write",
            "description": "Write content to a memory file. Creates the file if it doesn't exist.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path within memory directory (e.g. 'SOUL.md', 'USER.md', 'MEMORY.md', 'memory/2026-03-02.md')",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    ]
    agent.tool_functions = {
        "read": lambda **kw: "original",
        "write": lambda **kw: "original",
    }

    messages = _build_rich_conversation()

    with patch("app.services.memory_service._memory_dir", return_value=memory_root):
        await agent._run_memory_flush_turn(messages)

    # Check what was written
    agent_dir = memory_root / "agents" / agent_id
    print(f"\n=== Memory files in {agent_dir} ===")

    all_files = sorted(agent_dir.rglob("*.md")) if agent_dir.exists() else []
    for f in all_files:
        rel = f.relative_to(agent_dir)
        content = f.read_text(encoding="utf-8")
        print(f"\n--- {rel} ({len(content)} chars) ---")
        print(content[:500])
        if len(content) > 500:
            print(f"  ... ({len(content) - 500} more chars)")

    # At least some files should have been created
    assert len(all_files) > 0, "Flush turn produced no memory files at all"

    # Report which files were written
    written_names = set()
    for f in all_files:
        rel = f.relative_to(agent_dir)
        written_names.add(str(rel))

    for name in ["SOUL.md", "USER.md", "MEMORY.md"]:
        status = "WRITTEN" if name in written_names else "MISSING"
        print(f"  {name}: {status}")

    # Check for daily log
    memory_subdir = agent_dir / "memory"
    daily_logs = list(memory_subdir.glob("*.md")) if memory_subdir.exists() else []
    print(f"  Daily logs: {len(daily_logs)} files")

    # Concatenate all written content for keyword checks
    all_content = "\n".join(f.read_text(encoding="utf-8") for f in all_files)

    # The conversation had persona/preference/fact signals — verify they were captured
    # At minimum, we need 2+ files (LLM should split by category)
    assert len(all_files) >= 2, (
        f"Expected at least 2 memory files (SOUL + USER + daily log etc.), "
        f"got {len(all_files)}: {[str(f.relative_to(agent_dir)) for f in all_files]}"
    )

    # Verify content quality: key facts from conversation should appear somewhere
    content_lower = all_content.lower()
    assert "alex" in content_lower, "User name 'Alex' not found in any memory file"
    assert "python" in content_lower or "fastapi" in content_lower, (
        "User's tech preferences (Python/FastAPI) not found in any memory file"
    )
    assert "grpc" in content_lower or "migration" in content_lower, (
        "Architectural decision (gRPC migration) not found in any memory file"
    )
