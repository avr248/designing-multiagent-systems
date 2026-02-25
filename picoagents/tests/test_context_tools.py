"""Tests for context engineering tools.

This module tests:
1. TaskTool - Sub-agent spawning with context isolation
2. TodoWriteTool / TodoReadTool - Task tracking with persistence
3. SkillsTool - Progressive disclosure of domain expertise
4. MultiEditTool - Atomic multi-file edits

These tools implement patterns from Anthropic's context engineering research.
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src")
)

from picoagents.tools._context_tools import (
    MultiEditTool,
    SkillsTool,
    TaskTool,
    TodoListSessionsTool,
    TodoReadTool,
    TodoWriteTool,
    _get_skill_body,
    _parse_skill_frontmatter,
    create_context_engineering_tools,
    create_multi_edit_tool,
    create_skills_tool,
    create_task_tool,
    create_todo_tools,
    get_current_session_id,
    list_todo_sessions,
    set_session_id,
    set_todo_path,
)
from picoagents.types import ToolResult


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_todo_path(temp_dir):
    """Set up a temporary todo path for testing."""
    todo_path = temp_dir / "todos.json"
    set_todo_path(todo_path)
    set_session_id("test_session_001")  # Use consistent session for tests
    yield todo_path
    # Reset to default after test
    set_todo_path(None)
    set_session_id(None)


@pytest.fixture
def temp_skills_dir(temp_dir):
    """Create a temporary skills directory with sample skills."""
    skills_dir = temp_dir / "skills"
    skills_dir.mkdir()

    # Create a sample skill
    skill1_dir = skills_dir / "python-testing"
    skill1_dir.mkdir()
    (skill1_dir / "SKILL.md").write_text("""---
name: python-testing
description: Best practices for Python testing with pytest
triggers: test, pytest, unittest
---

# Python Testing Skill

## Guidelines

1. Use pytest fixtures for setup/teardown
2. Use parametrized tests for multiple inputs
3. Mock external dependencies

## Example

```python
import pytest

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_example(sample_data):
    assert sample_data["key"] == "value"
```
""")

    # Create another skill
    skill2_dir = skills_dir / "git-workflow"
    skill2_dir.mkdir()
    (skill2_dir / "SKILL.md").write_text("""---
name: git-workflow
description: Git workflow and branching strategies
triggers: git, branch, commit, merge
---

# Git Workflow

Use feature branches and pull requests.
""")

    return skills_dir


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for editing tests."""
    file_path = temp_dir / "sample.py"
    file_path.write_text("""def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")

def main():
    hello()
    goodbye()
""")
    return file_path


# =============================================================================
# TodoWriteTool Tests
# =============================================================================


class TestTodoWriteTool:
    """Tests for TodoWriteTool."""

    @pytest.mark.asyncio
    async def test_create_basic_todos(self, temp_todo_path):
        """Test creating a basic todo list."""
        tool = TodoWriteTool()

        todos = [
            {"content": "Write tests", "status": "pending", "activeForm": "Writing tests"},
            {"content": "Run linter", "status": "pending", "activeForm": "Running linter"},
        ]

        result = await tool.execute({"todos": todos})

        assert result.success is True
        assert "2 pending" in result.result
        assert "0 completed" in result.result

        # Verify persistence (new format has metadata wrapper)
        saved = json.loads(temp_todo_path.read_text())
        # New format: {"session_id": ..., "todos": [...]}
        if isinstance(saved, dict):
            assert len(saved.get("todos", [])) == 2
        else:
            assert len(saved) == 2

    @pytest.mark.asyncio
    async def test_update_todo_status(self, temp_todo_path):
        """Test updating todo status."""
        tool = TodoWriteTool()

        # Create initial todos
        todos = [
            {"content": "Task 1", "status": "completed", "activeForm": "Completing Task 1"},
            {"content": "Task 2", "status": "in_progress", "activeForm": "Working on Task 2"},
            {"content": "Task 3", "status": "pending", "activeForm": "Starting Task 3"},
        ]

        result = await tool.execute({"todos": todos})

        assert result.success is True
        assert "1 completed" in result.result
        assert "1 in progress" in result.result
        assert "1 pending" in result.result
        assert "Working on Task 2" in result.result  # Current task

    @pytest.mark.asyncio
    async def test_only_one_in_progress_allowed(self, temp_todo_path):
        """Test that only one task can be in_progress."""
        tool = TodoWriteTool()

        todos = [
            {"content": "Task 1", "status": "in_progress", "activeForm": "Working on Task 1"},
            {"content": "Task 2", "status": "in_progress", "activeForm": "Working on Task 2"},
        ]

        result = await tool.execute({"todos": todos})

        assert result.success is False
        assert "Only one allowed" in result.error

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, temp_todo_path):
        """Test validation of required fields."""
        tool = TodoWriteTool()

        # Missing content
        result = await tool.execute({"todos": [{"status": "pending", "activeForm": "Test"}]})
        assert result.success is False
        assert "missing 'content'" in result.error

        # Missing status
        result = await tool.execute({"todos": [{"content": "Test", "activeForm": "Test"}]})
        assert result.success is False
        assert "missing 'status'" in result.error

        # Missing activeForm
        result = await tool.execute({"todos": [{"content": "Test", "status": "pending"}]})
        assert result.success is False
        assert "missing 'activeForm'" in result.error

    @pytest.mark.asyncio
    async def test_invalid_status(self, temp_todo_path):
        """Test validation of status values."""
        tool = TodoWriteTool()

        result = await tool.execute({
            "todos": [{"content": "Test", "status": "invalid", "activeForm": "Test"}]
        })

        assert result.success is False
        assert "invalid status" in result.error


# =============================================================================
# TodoReadTool Tests
# =============================================================================


class TestTodoReadTool:
    """Tests for TodoReadTool."""

    @pytest.mark.asyncio
    async def test_read_empty_todos(self, temp_todo_path):
        """Test reading when no todos exist."""
        tool = TodoReadTool()

        result = await tool.execute({})

        assert result.success is True
        assert "No todos" in result.result

    @pytest.mark.asyncio
    async def test_read_existing_todos(self, temp_todo_path):
        """Test reading existing todos."""
        # Create todos first
        write_tool = TodoWriteTool()
        await write_tool.execute({
            "todos": [
                {"content": "Task 1", "status": "completed", "activeForm": "Task 1"},
                {"content": "Task 2", "status": "in_progress", "activeForm": "Task 2"},
                {"content": "Task 3", "status": "pending", "activeForm": "Task 3"},
            ]
        })

        # Read them back
        read_tool = TodoReadTool()
        result = await read_tool.execute({})

        assert result.success is True
        assert "Progress: 1/3" in result.result
        assert "✓ Task 1" in result.result  # completed
        assert "→ Task 2" in result.result  # in_progress
        assert "○ Task 3" in result.result  # pending

    @pytest.mark.asyncio
    async def test_read_returns_metadata(self, temp_todo_path):
        """Test that read returns metadata."""
        write_tool = TodoWriteTool()
        await write_tool.execute({
            "todos": [
                {"content": "Task 1", "status": "completed", "activeForm": "Task 1"},
                {"content": "Task 2", "status": "pending", "activeForm": "Task 2"},
            ]
        })

        read_tool = TodoReadTool()
        result = await read_tool.execute({})

        assert result.metadata is not None
        assert result.metadata["completed"] == 1
        assert result.metadata["total"] == 2


# =============================================================================
# TodoListSessionsTool and Session Management Tests
# =============================================================================


class TestTodoSessions:
    """Tests for todo session management."""

    @pytest.fixture
    def session_dir(self, temp_dir):
        """Set up session-based storage for testing."""
        todos_dir = temp_dir / "todos"
        todos_dir.mkdir(parents=True, exist_ok=True)

        # Reset global state
        set_todo_path(None)
        set_session_id(None)

        # Create some mock sessions
        session1 = todos_dir / "session_2024-02-01_abc12345.json"
        session1.write_text(json.dumps({
            "session_id": "2024-02-01_abc12345",
            "todos": [
                {"content": "Old task 1", "status": "completed", "activeForm": "Task 1"},
                {"content": "Old task 2", "status": "completed", "activeForm": "Task 2"},
            ]
        }))

        session2 = todos_dir / "session_2024-02-02_def67890.json"
        session2.write_text(json.dumps({
            "session_id": "2024-02-02_def67890",
            "todos": [
                {"content": "Recent task", "status": "in_progress", "activeForm": "Task"},
            ]
        }))

        # Monkey-patch _get_workspace to use temp_dir
        import picoagents.tools._context_tools as ctx
        original_get_workspace = ctx._get_workspace
        ctx._get_workspace = lambda: temp_dir

        yield todos_dir

        # Restore
        ctx._get_workspace = original_get_workspace
        set_todo_path(None)
        set_session_id(None)

    def test_list_sessions(self, session_dir):
        """Test listing all sessions."""
        sessions = list_todo_sessions()

        assert len(sessions) >= 2
        session_ids = [s["session_id"] for s in sessions]
        assert "2024-02-01_abc12345" in session_ids
        assert "2024-02-02_def67890" in session_ids

    def test_session_id_generation(self, temp_dir):
        """Test that session IDs are generated correctly."""
        set_session_id(None)  # Reset
        session_id = get_current_session_id()

        # Should be date + short UUID format
        assert "_" in session_id
        date_part, uuid_part = session_id.split("_", 1)

        # Date should be YYYY-MM-DD format
        from datetime import datetime
        datetime.strptime(date_part, "%Y-%m-%d")

        # UUID part should be 8 hex chars
        assert len(uuid_part) == 8
        int(uuid_part, 16)  # Should be valid hex

    def test_set_custom_session_id(self, temp_dir):
        """Test setting a custom session ID."""
        set_session_id("my_custom_session")
        assert get_current_session_id() == "my_custom_session"
        set_session_id(None)

    @pytest.mark.asyncio
    async def test_sessions_tool(self, session_dir):
        """Test TodoListSessionsTool."""
        tool = TodoListSessionsTool()
        result = await tool.execute({})

        assert result.success is True
        assert "session" in result.result.lower()
        assert "2024-02-01_abc12345" in result.result or "abc12345" in result.result

    @pytest.mark.asyncio
    async def test_read_specific_session(self, session_dir):
        """Test reading todos from a specific session."""
        set_session_id("new_session")  # Current session is different

        read_tool = TodoReadTool()
        result = await read_tool.execute({"session_id": "2024-02-01_abc12345"})

        # Should read from old session
        assert result.success is True
        assert "Old task 1" in result.result or "2/2" in result.result

        # Current session should be restored
        assert get_current_session_id() == "new_session"


# =============================================================================
# SkillsTool Tests
# =============================================================================


class TestSkillsTool:
    """Tests for SkillsTool."""

    @pytest.mark.asyncio
    async def test_list_skills(self, temp_skills_dir):
        """Test listing available skills."""
        tool = SkillsTool(project_path=temp_skills_dir)

        result = await tool.execute({"action": "list"})

        assert result.success is True
        assert "python-testing" in result.result
        assert "git-workflow" in result.result
        assert "Best practices for Python testing" in result.result

    @pytest.mark.asyncio
    async def test_list_no_skills(self, temp_dir):
        """Test listing when no skills exist."""
        empty_skills = temp_dir / "empty_skills"
        empty_skills.mkdir()

        tool = SkillsTool(project_path=empty_skills)
        result = await tool.execute({"action": "list"})

        assert result.success is True
        assert "No skills found" in result.result

    @pytest.mark.asyncio
    async def test_load_skill(self, temp_skills_dir):
        """Test loading a specific skill."""
        tool = SkillsTool(project_path=temp_skills_dir)

        result = await tool.execute({"action": "load", "name": "python-testing"})

        assert result.success is True
        assert "Python Testing Skill" in result.result
        assert "pytest fixtures" in result.result
        assert "@pytest.fixture" in result.result  # Example code

    @pytest.mark.asyncio
    async def test_load_nonexistent_skill(self, temp_skills_dir):
        """Test loading a skill that doesn't exist."""
        tool = SkillsTool(project_path=temp_skills_dir)

        result = await tool.execute({"action": "load", "name": "nonexistent"})

        assert result.success is False
        assert "not found" in result.error
        assert "python-testing" in result.error  # Suggests available skills

    @pytest.mark.asyncio
    async def test_load_missing_name(self, temp_skills_dir):
        """Test that load requires name parameter."""
        tool = SkillsTool(project_path=temp_skills_dir)

        result = await tool.execute({"action": "load"})

        assert result.success is False
        assert "'name' parameter is required" in result.error

    @pytest.mark.asyncio
    async def test_invalid_action(self, temp_skills_dir):
        """Test invalid action parameter."""
        tool = SkillsTool(project_path=temp_skills_dir)

        result = await tool.execute({"action": "invalid"})

        assert result.success is False
        assert "Unknown action" in result.error

    def test_system_prompt_section_with_skills(self, temp_skills_dir):
        """Test that get_system_prompt_section returns skill metadata."""
        tool = SkillsTool(project_path=temp_skills_dir)

        section = tool.get_system_prompt_section()

        assert "Available Skills" in section
        assert "python-testing" in section
        assert "git-workflow" in section
        assert "Best practices for Python testing" in section
        assert "Git workflow and branching" in section
        assert "skills(action='load'" in section

    def test_system_prompt_section_no_skills(self, temp_dir):
        """Test that get_system_prompt_section returns empty string when no skills."""
        empty_skills = temp_dir / "empty_skills"
        empty_skills.mkdir()

        tool = SkillsTool(project_path=empty_skills)

        section = tool.get_system_prompt_section()

        assert section == ""

    def test_system_prompt_section_no_full_content(self, temp_skills_dir):
        """Test that system prompt section contains only metadata, not full skill body."""
        tool = SkillsTool(project_path=temp_skills_dir)

        section = tool.get_system_prompt_section()

        # Should have descriptions but NOT the full skill body
        assert "python-testing" in section
        assert "pytest fixtures" not in section  # Full body content
        assert "@pytest.fixture" not in section  # Example code from skill


class TestSkillFrontmatterParsing:
    """Tests for SKILL.md frontmatter parsing."""

    def test_parse_frontmatter(self):
        """Test parsing YAML frontmatter."""
        content = """---
name: test-skill
description: A test skill
triggers: test, testing
---

# Content here
"""
        result = _parse_skill_frontmatter(content)

        assert result["name"] == "test-skill"
        assert result["description"] == "A test skill"
        assert result["triggers"] == "test, testing"

    def test_parse_no_frontmatter(self):
        """Test parsing content without frontmatter."""
        content = "# Just content\nNo frontmatter here."
        result = _parse_skill_frontmatter(content)
        assert result == {}

    def test_get_skill_body(self):
        """Test extracting body after frontmatter."""
        content = """---
name: test
---

# The Body

Content here.
"""
        body = _get_skill_body(content)
        assert "# The Body" in body
        assert "Content here." in body
        assert "name:" not in body

    def test_get_skill_body_no_frontmatter(self):
        """Test extracting body when no frontmatter."""
        content = "# Just content"
        body = _get_skill_body(content)
        assert body == content


# =============================================================================
# MultiEditTool Tests
# =============================================================================


class TestMultiEditTool:
    """Tests for MultiEditTool."""

    @pytest.mark.asyncio
    async def test_single_edit(self, sample_file, temp_dir):
        """Test a single edit."""
        tool = MultiEditTool(workspace=temp_dir)

        result = await tool.execute({
            "path": str(sample_file),
            "edits": [
                {"old_string": "Hello, World!", "new_string": "Hi there!"}
            ]
        })

        assert result.success is True
        assert "Successfully applied 1 edit" in result.result

        content = sample_file.read_text()
        assert "Hi there!" in content
        assert "Hello, World!" not in content

    @pytest.mark.asyncio
    async def test_multiple_edits(self, sample_file, temp_dir):
        """Test multiple edits applied atomically."""
        tool = MultiEditTool(workspace=temp_dir)

        result = await tool.execute({
            "path": str(sample_file),
            "edits": [
                {"old_string": "Hello, World!", "new_string": "Hi there!"},
                {"old_string": "Goodbye!", "new_string": "See ya!"},
            ]
        })

        assert result.success is True
        assert "Successfully applied 2 edit" in result.result

        content = sample_file.read_text()
        assert "Hi there!" in content
        assert "See ya!" in content

    @pytest.mark.asyncio
    async def test_edit_not_found_rollback(self, sample_file, temp_dir):
        """Test that file is unchanged when edit not found."""
        tool = MultiEditTool(workspace=temp_dir)

        original = sample_file.read_text()

        result = await tool.execute({
            "path": str(sample_file),
            "edits": [
                {"old_string": "Hello, World!", "new_string": "Hi!"},
                {"old_string": "NONEXISTENT STRING", "new_string": "test"},
            ]
        })

        assert result.success is False
        assert "could not find text" in result.error
        assert "atomic rollback" in result.error

        # File should be unchanged
        assert sample_file.read_text() == original

    @pytest.mark.asyncio
    async def test_edit_multiple_occurrences_fails(self, temp_dir):
        """Test that edit fails if old_string appears multiple times."""
        file_path = temp_dir / "duplicate.py"
        file_path.write_text("hello\nhello\nworld")

        tool = MultiEditTool(workspace=temp_dir)

        result = await tool.execute({
            "path": str(file_path),
            "edits": [
                {"old_string": "hello", "new_string": "hi"}
            ]
        })

        assert result.success is False
        assert "2 occurrences" in result.error
        assert "must be unique" in result.error

    @pytest.mark.asyncio
    async def test_file_not_found(self, temp_dir):
        """Test error when file doesn't exist."""
        tool = MultiEditTool(workspace=temp_dir)

        result = await tool.execute({
            "path": str(temp_dir / "nonexistent.py"),
            "edits": [
                {"old_string": "test", "new_string": "test2"}
            ]
        })

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_missing_parameters(self, temp_dir):
        """Test validation of required parameters."""
        tool = MultiEditTool(workspace=temp_dir)

        # Missing path
        result = await tool.execute({"edits": []})
        assert result.success is False
        assert "'path' is required" in result.error

        # Missing edits
        result = await tool.execute({"path": "/some/path"})
        assert result.success is False
        assert "'edits' list is required" in result.error

    @pytest.mark.asyncio
    async def test_sequential_edit_application(self, temp_dir):
        """Test that edits are applied sequentially."""
        file_path = temp_dir / "sequential.txt"
        file_path.write_text("AAA")

        tool = MultiEditTool(workspace=temp_dir)

        # First edit changes AAA to BBB, second changes BBB to CCC
        result = await tool.execute({
            "path": str(file_path),
            "edits": [
                {"old_string": "AAA", "new_string": "BBB"},
                {"old_string": "BBB", "new_string": "CCC"},
            ]
        })

        assert result.success is True
        assert file_path.read_text() == "CCC"


# =============================================================================
# TaskTool Tests
# =============================================================================


class TestTaskTool:
    """Tests for TaskTool."""

    def test_task_tool_creation(self):
        """Test creating a TaskTool."""
        tool = TaskTool()

        assert tool.name == "task"
        assert "sub-agent" in tool.description.lower()

    def test_task_tool_with_coordinator(self):
        """Test creating a TaskTool with coordinator."""
        mock_agent = MagicMock()
        mock_agent.tools = []

        tool = TaskTool(coordinator=mock_agent)

        assert tool.coordinator is mock_agent

    def test_parameters_schema(self):
        """Test the parameters schema."""
        tool = TaskTool()
        params = tool.parameters

        assert params["type"] == "object"
        assert "prompt" in params["properties"]
        assert "description" in params["properties"]
        assert "agent_type" in params["properties"]
        assert params["properties"]["agent_type"]["enum"] == ["explore", "research", "general"]

    @pytest.mark.asyncio
    async def test_execute_no_model_client(self):
        """Test that execute fails without model client."""
        tool = TaskTool()

        result = await tool.execute({
            "prompt": "Test task",
            "description": "Test"
        })

        assert result.success is False
        assert "No model client" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_prompt(self):
        """Test that execute fails without prompt."""
        mock_client = MagicMock()
        tool = TaskTool(model_client=mock_client)

        result = await tool.execute({
            "description": "Test"
        })

        assert result.success is False
        assert "'prompt' parameter is required" in result.error


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_task_tool(self):
        """Test create_task_tool factory."""
        tool = create_task_tool(token_budget=30_000, max_iterations=10)

        assert isinstance(tool, TaskTool)
        assert tool.token_budget == 30_000
        assert tool.max_iterations == 10

    def test_create_todo_tools(self):
        """Test create_todo_tools factory."""
        tools = create_todo_tools()

        assert len(tools) == 2
        assert any(isinstance(t, TodoWriteTool) for t in tools)
        assert any(isinstance(t, TodoReadTool) for t in tools)

    def test_create_skills_tool(self, temp_skills_dir):
        """Test create_skills_tool factory."""
        tool = create_skills_tool(project_path=temp_skills_dir)

        assert isinstance(tool, SkillsTool)
        assert temp_skills_dir in tool.skill_paths

    def test_create_multi_edit_tool(self, temp_dir):
        """Test create_multi_edit_tool factory."""
        tool = create_multi_edit_tool(workspace=temp_dir)

        assert isinstance(tool, MultiEditTool)
        assert tool.workspace == temp_dir

    def test_create_context_engineering_tools(self, temp_skills_dir, temp_dir):
        """Test create_context_engineering_tools factory."""
        tools = create_context_engineering_tools(
            skills_path=temp_skills_dir,
            workspace=temp_dir,
        )

        # Should have TaskTool, TodoWriteTool, TodoReadTool, SkillsTool, MultiEditTool
        assert len(tools) == 5

        tool_names = [t.name for t in tools]
        assert "task" in tool_names
        assert "todo_write" in tool_names
        assert "todo_read" in tool_names
        assert "skills" in tool_names
        assert "multi_edit" in tool_names


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for context engineering tools."""

    @pytest.mark.asyncio
    async def test_todo_workflow(self, temp_todo_path):
        """Test a complete todo workflow."""
        write_tool = TodoWriteTool()
        read_tool = TodoReadTool()

        # Step 1: Create initial todos
        await write_tool.execute({
            "todos": [
                {"content": "Research topic", "status": "in_progress", "activeForm": "Researching"},
                {"content": "Write draft", "status": "pending", "activeForm": "Writing draft"},
                {"content": "Review", "status": "pending", "activeForm": "Reviewing"},
            ]
        })

        # Step 2: Check progress
        result = await read_tool.execute({})
        assert "Progress: 0/3" in result.result
        assert "→ Research topic" in result.result

        # Step 3: Complete first task, start second
        await write_tool.execute({
            "todos": [
                {"content": "Research topic", "status": "completed", "activeForm": "Researching"},
                {"content": "Write draft", "status": "in_progress", "activeForm": "Writing draft"},
                {"content": "Review", "status": "pending", "activeForm": "Reviewing"},
            ]
        })

        # Step 4: Verify update
        result = await read_tool.execute({})
        assert "Progress: 1/3" in result.result
        assert "✓ Research topic" in result.result
        assert "→ Write draft" in result.result

    @pytest.mark.asyncio
    async def test_skills_workflow(self, temp_skills_dir):
        """Test skills discovery and loading workflow."""
        tool = SkillsTool(project_path=temp_skills_dir)

        # Step 1: List skills (progressive disclosure - summaries only)
        result = await tool.execute({"action": "list"})
        assert "python-testing" in result.result
        assert "pytest fixtures" not in result.result  # Full content not shown

        # Step 2: Load specific skill when needed
        result = await tool.execute({"action": "load", "name": "python-testing"})
        assert "pytest fixtures" in result.result  # Now we get full content

    @pytest.mark.asyncio
    async def test_multi_edit_workflow(self, temp_dir):
        """Test multi-edit for refactoring."""
        # Create a file needing refactoring
        # Note: Each string to replace must be unique in the file
        file_path = temp_dir / "refactor_me.py"
        file_path.write_text("""
def calculate(x, y):
    sum_result = x + y
    return sum_result

def process(x, y):
    computed_value = calculate(x, y)
    return computed_value * 2
""")

        tool = MultiEditTool(workspace=temp_dir)

        # Refactor: rename variables
        result = await tool.execute({
            "path": str(file_path),
            "edits": [
                {"old_string": "sum_result = x + y", "new_string": "total = x + y"},
                {"old_string": "return sum_result", "new_string": "return total"},
                {"old_string": "computed_value = calculate", "new_string": "output = calculate"},
                {"old_string": "return computed_value * 2", "new_string": "return output * 2"},
            ]
        })

        assert result.success is True
        assert "4 edit" in result.result

        content = file_path.read_text()
        assert "total = x + y" in content
        assert "return total" in content
        assert "output = calculate" in content
        assert "return output * 2" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
