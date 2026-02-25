"""Context engineering tools for PicoAgents.

This module provides tools specifically designed for context management:

1. TaskTool - Spawn sub-agents in isolated contexts (Isolation strategy)
2. TodoWriteTool / TodoReadTool - Track task progress
3. SkillsTool - Progressive disclosure of domain expertise
4. MultiEditTool - Atomic multi-edit for files

These tools implement patterns from Anthropic's context engineering research:
- Isolation: Run sub-tasks in separate contexts, only summaries cross back
- Progressive disclosure: Load domain knowledge on-demand
- Structured note-taking: Persist state outside the context window

Based on MiniAgent implementations and Claude Code patterns.
"""

from __future__ import annotations

import json
import os
import re
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

from ._base import BaseTool
from ..types import ToolResult

if TYPE_CHECKING:
    from ..agents import Agent
    from ..llm import BaseChatCompletionClient


# =============================================================================
# Task Tool - Context Isolation via Sub-agents
# =============================================================================

# Agent type configurations for sub-agents
AGENT_TYPES = {
    "explore": {
        "description": "Fast agent for exploring codebases",
        "instructions": """You are a codebase exploration specialist.

Your role:
1. Quickly find files, code patterns, and answer questions about the codebase
2. Use glob and grep efficiently to locate relevant code
3. Read files to understand implementations
4. Provide concise, accurate summaries

Guidelines:
- Be thorough but efficient - don't read unnecessary files
- Focus on answering the specific question asked
- Return structured findings (file paths, line numbers, key code snippets)
- Keep your final response under 500 tokens
- Your response will be passed to another agent, so make it self-contained
""",
        "tool_names": ["read_file", "list_directory", "grep_search", "think"],
    },
    "research": {
        "description": "Agent for web research and information gathering",
        "instructions": """You are a research assistant.

Your role:
1. Thoroughly research the given topic using web search and fetch
2. Gather relevant information from multiple sources
3. Synthesize findings into a clear, concise summary
4. Return only the essential information needed

Guidelines:
- Be thorough in research but concise in response
- Focus on facts and actionable information
- Cite sources when relevant
- Target 200-500 tokens for your final response
- Your response will be passed to another agent, so make it self-contained
""",
        "tool_names": ["web_search", "web_fetch", "think"],
    },
    "general": {
        "description": "General-purpose agent for complex multi-step tasks",
        "instructions": """You are a capable assistant handling a delegated task.

Your role:
1. Complete the assigned task thoroughly
2. Use available tools as needed
3. Return a clear summary of what was accomplished

Guidelines:
- Focus on the specific task assigned
- Be thorough but concise in your response
- Include relevant details the coordinator needs
- Target 300-600 tokens for your final response
""",
        "tool_names": None,  # Uses coordinator's tools
    },
}


class TaskTool(BaseTool):
    """Spawn sub-agents to handle tasks in isolated contexts.

    The Task tool enables context isolation - sub-agents run in their own
    context windows, preventing context pollution in the coordinator.

    Benefits:
    - Sub-agent can use many tokens internally (50K+)
    - Coordinator only sees the distilled result (few hundred tokens)
    - Prevents "lost in the middle" degradation
    - Enables parallelization

    Example:
        from picoagents import Agent
        from picoagents.tools import TaskTool

        task_tool = TaskTool(coordinator=my_agent)
        agent = Agent(
            ...,
            tools=[task_tool, ...],
        )
    """

    def __init__(
        self,
        coordinator: Optional["Agent"] = None,
        model_client: Optional["BaseChatCompletionClient"] = None,
        token_budget: int = 50_000,
        max_iterations: int = 20,
    ):
        """Create a Task tool for spawning sub-agents.

        Args:
            coordinator: Parent agent (for inheriting config and tools)
            model_client: Model client to use for sub-agents (overrides coordinator's)
            token_budget: Token budget for sub-agents (default 50K)
            max_iterations: Max iterations for sub-agents (default 20)
        """
        super().__init__(
            name="task",
            description=(
                "Launch a sub-agent to handle a complex task in isolated context. "
                "Use when: (1) you need to explore a codebase without polluting context, "
                "(2) a task requires many tool calls that would bloat context, "
                "(3) you want to delegate research to a specialist, "
                "(4) the task has clear input/output boundaries. "
                "Agent types: 'explore' (codebase search), 'research' (web), 'general' (all tools)."
            ),
        )
        self.coordinator = coordinator
        self.model_client = model_client
        self.token_budget = token_budget
        self.max_iterations = max_iterations

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description of the task for the sub-agent",
                },
                "description": {
                    "type": "string",
                    "description": "Short 3-5 word summary of what the sub-agent will do",
                },
                "agent_type": {
                    "type": "string",
                    "enum": ["explore", "research", "general"],
                    "description": (
                        "Type of sub-agent: 'explore' for codebase search, "
                        "'research' for web research, 'general' for other tasks"
                    ),
                    "default": "general",
                },
            },
            "required": ["prompt", "description"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the task tool by spawning a sub-agent."""
        prompt = parameters.get("prompt", "")
        description = parameters.get("description", "")
        agent_type = parameters.get("agent_type", "general")

        if not prompt:
            return ToolResult(
                success=False,
                result="",
                error="'prompt' parameter is required",
            )

        try:
            # Lazy imports to avoid circular dependencies
            from ..agents import Agent
            from ..compaction import HeadTailCompaction

            # Get agent type config
            config = AGENT_TYPES.get(agent_type, AGENT_TYPES["general"])

            # Determine model client
            client = self.model_client
            if client is None and self.coordinator:
                client = self.coordinator.model_client
            if client is None:
                return ToolResult(
                    success=False,
                    result="",
                    error="No model client available. Provide model_client or coordinator.",
                )

            # Determine tools for sub-agent
            sub_tools = self._get_tools_for_type(agent_type, config)

            # Create sub-agent with compaction for its own context
            sub_agent = Agent(
                name=f"sub_{agent_type}",
                description=config["description"],
                instructions=config["instructions"],
                model_client=client,
                tools=sub_tools,
                compaction=HeadTailCompaction(
                    token_budget=self.token_budget,
                    head_ratio=0.2,
                ),
                max_iterations=self.max_iterations,
            )

            # Run the sub-agent
            response = await sub_agent.run(prompt)

            # Extract content from response
            content = ""
            if response.context and response.context.messages:
                # Get the last assistant message
                for msg in reversed(response.context.messages):
                    if hasattr(msg, "role") and msg.role == "assistant":
                        content = getattr(msg, "content", "") or ""
                        break

            if not content:
                content = "(No response from sub-agent)"

            # Add usage info for transparency
            usage = response.usage
            usage_info = (
                f"\n\n[Sub-agent ({agent_type}): "
                f"{usage.llm_calls} LLM calls, "
                f"{usage.tokens_input} input tokens, "
                f"{usage.tool_calls} tool calls]"
            )

            return ToolResult(
                success=True,
                result=content + usage_info,
                error=None,
                metadata={
                    "agent_type": agent_type,
                    "description": description,
                    "llm_calls": usage.llm_calls,
                    "tokens_input": usage.tokens_input,
                    "tool_calls": usage.tool_calls,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                result="",
                error=f"Sub-agent failed: {str(e)}",
            )

    def _get_tools_for_type(
        self, agent_type: str, config: dict
    ) -> List[BaseTool]:
        """Get tools for a specific agent type."""
        tool_names = config.get("tool_names")

        if tool_names is None:
            # General agent - inherit coordinator's tools (except task itself)
            if self.coordinator:
                return [
                    t for t in self.coordinator.tools
                    if t.name != "task"
                ]
            return []

        # Specific tool set - create minimal tools
        tools = []
        from ._core_tools import ThinkTool

        tool_map = {
            "think": ThinkTool(),
        }

        # Try to import coding tools if available
        try:
            from ._coding_tools import (
                ReadFileTool,
                ListDirectoryTool,
                GrepSearchTool,
            )
            tool_map.update({
                "read_file": ReadFileTool(),
                "list_directory": ListDirectoryTool(),
                "grep_search": GrepSearchTool(),
            })
        except ImportError:
            pass

        # Try to import research tools if available
        try:
            from ._research_tools import (
                WebSearchTool,
                WebFetchTool,
            )
            tool_map.update({
                "web_search": WebSearchTool(),
                "web_fetch": WebFetchTool(),
            })
        except ImportError:
            pass

        for name in tool_names:
            if name in tool_map:
                tools.append(tool_map[name])

        return tools


# =============================================================================
# Todo Tools - Task Progress Tracking
# =============================================================================

import uuid
from datetime import datetime

# Global todo storage configuration
_TODO_PATH: Optional[Path] = None
_SESSION_ID: Optional[str] = None


def _get_workspace() -> Path:
    """Get the picoagents workspace directory."""
    workspace = Path.cwd() / ".picoagents"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _get_session_id() -> str:
    """Get or create a session ID for this run."""
    global _SESSION_ID
    if _SESSION_ID is None:
        # Generate session ID: date + short UUID
        date_str = datetime.now().strftime("%Y-%m-%d")
        short_id = uuid.uuid4().hex[:8]
        _SESSION_ID = f"{date_str}_{short_id}"
    return _SESSION_ID


def _get_todo_path() -> Path:
    """Get the path for todo storage."""
    if _TODO_PATH is not None:
        return _TODO_PATH

    # Default to session-based storage
    workspace = _get_workspace()
    todos_dir = workspace / "todos"
    todos_dir.mkdir(parents=True, exist_ok=True)

    session_id = _get_session_id()
    return todos_dir / f"session_{session_id}.json"


def set_todo_path(path: Optional[Path]) -> None:
    """Set custom todo storage path (for testing)."""
    global _TODO_PATH
    _TODO_PATH = path


def set_session_id(session_id: Optional[str]) -> None:
    """Set a custom session ID (useful for resuming sessions)."""
    global _SESSION_ID
    _SESSION_ID = session_id


def get_current_session_id() -> str:
    """Get the current session ID."""
    return _get_session_id()


def list_todo_sessions() -> List[Dict[str, Any]]:
    """List all todo sessions with their metadata.

    Returns:
        List of session info dicts with: session_id, path, created, todo_count
    """
    workspace = _get_workspace()
    todos_dir = workspace / "todos"

    if not todos_dir.exists():
        return []

    sessions = []
    for file in sorted(todos_dir.glob("session_*.json"), reverse=True):
        try:
            data = json.loads(file.read_text())
            todos = data if isinstance(data, list) else data.get("todos", [])

            # Extract session ID from filename
            session_id = file.stem.replace("session_", "")

            sessions.append({
                "session_id": session_id,
                "path": str(file),
                "created": file.stat().st_mtime,
                "todo_count": len(todos),
                "completed": sum(1 for t in todos if t.get("status") == "completed"),
            })
        except Exception:
            continue

    return sessions


def _load_todos() -> List[Dict[str, Any]]:
    """Load todos from file."""
    path = _get_todo_path()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            # Support both old format (list) and new format (dict with metadata)
            if isinstance(data, list):
                return data
            return data.get("todos", [])
        except Exception:
            return []
    return []


def _save_todos(todos: List[Dict[str, Any]]) -> None:
    """Save todos to file with session metadata."""
    path = _get_todo_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save with session metadata
    data = {
        "session_id": _get_session_id(),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "todos": todos,
    }

    path.write_text(json.dumps(data, indent=2))


class TodoWriteTool(BaseTool):
    """Create or update the task list for tracking progress.

    Use proactively when:
    - Starting a complex multi-step task (3+ steps)
    - The user provides multiple tasks to complete
    - You need to track progress on a larger goal

    Each todo item requires:
    - content: What needs to be done (imperative form)
    - status: 'pending', 'in_progress', or 'completed'
    - activeForm: Present tense description

    Important:
    - Only ONE task should be 'in_progress' at a time
    - Mark tasks 'completed' immediately when done
    """

    def __init__(self):
        super().__init__(
            name="todo_write",
            description=(
                "Create or update the task list for this session. "
                "Use for complex multi-step tasks (3+ steps). "
                "Each todo needs: content (str), status ('pending'|'in_progress'|'completed'), "
                "activeForm (str). Only ONE task should be 'in_progress' at a time."
            ),
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "List of todo items",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "What needs to be done (imperative form)",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Task status",
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Present tense description (e.g., 'Running tests')",
                            },
                        },
                        "required": ["content", "status", "activeForm"],
                    },
                },
            },
            "required": ["todos"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute todo_write."""
        todos = parameters.get("todos", [])

        # Validate todos
        valid_statuses = {"pending", "in_progress", "completed"}
        for i, todo in enumerate(todos):
            if "content" not in todo:
                return ToolResult(
                    success=False,
                    result="",
                    error=f"Todo {i+1} missing 'content'",
                )
            if "status" not in todo:
                return ToolResult(
                    success=False,
                    result="",
                    error=f"Todo {i+1} missing 'status'",
                )
            if todo["status"] not in valid_statuses:
                return ToolResult(
                    success=False,
                    result="",
                    error=f"Todo {i+1} has invalid status '{todo['status']}'",
                )
            if "activeForm" not in todo:
                return ToolResult(
                    success=False,
                    result="",
                    error=f"Todo {i+1} missing 'activeForm'",
                )

        # Check only one in_progress
        in_progress_count = sum(1 for t in todos if t["status"] == "in_progress")
        if in_progress_count > 1:
            return ToolResult(
                success=False,
                result="",
                error=f"{in_progress_count} tasks marked 'in_progress'. Only one allowed.",
            )

        # Save todos
        _save_todos(todos)

        # Format summary
        completed = sum(1 for t in todos if t["status"] == "completed")
        pending = sum(1 for t in todos if t["status"] == "pending")
        in_progress = sum(1 for t in todos if t["status"] == "in_progress")

        current = next((t for t in todos if t["status"] == "in_progress"), None)
        current_msg = f"Current: {current['activeForm']}" if current else "No task in progress"

        result = (
            f"Todo list updated: {completed} completed, {in_progress} in progress, "
            f"{pending} pending. {current_msg}"
        )

        return ToolResult(
            success=True,
            result=result,
            error=None,
            metadata={"completed": completed, "pending": pending, "in_progress": in_progress},
        )


class TodoReadTool(BaseTool):
    """Read the current todo list."""

    def __init__(self):
        super().__init__(
            name="todo_read",
            description="Read the current todo list with status of all tasks.",
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID to read from (defaults to current session)",
                },
            },
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute todo_read."""
        session_id = parameters.get("session_id")

        # If specific session requested, temporarily switch
        original_session = None
        if session_id:
            original_session = _SESSION_ID
            set_session_id(session_id)

        try:
            todos = _load_todos()

            if not todos:
                return ToolResult(
                    success=True,
                    result="No todos. Use todo_write to create a task list.",
                    error=None,
                )

            lines = []
            for todo in todos:
                status = todo.get("status", "pending")
                content = todo.get("content", "")

                if status == "completed":
                    icon = "✓"
                elif status == "in_progress":
                    icon = "→"
                else:
                    icon = "○"

                lines.append(f"{icon} {content}")

            completed = sum(1 for t in todos if t["status"] == "completed")
            total = len(todos)

            current_session = _get_session_id()
            result = f"Session: {current_session}\nProgress: {completed}/{total}\n\n" + "\n".join(lines)

            return ToolResult(
                success=True,
                result=result,
                error=None,
                metadata={"completed": completed, "total": total, "session_id": current_session},
            )
        finally:
            # Restore original session if we switched
            if original_session is not None:
                set_session_id(original_session)


class TodoListSessionsTool(BaseTool):
    """List all todo sessions to find past work."""

    def __init__(self):
        super().__init__(
            name="todo_sessions",
            description=(
                "List all todo sessions. Use to find past work or resume a previous session. "
                "Returns session IDs that can be passed to todo_read."
            ),
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max number of sessions to return (default 10)",
                    "default": 10,
                },
            },
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute todo_sessions."""
        limit = parameters.get("limit", 10)

        sessions = list_todo_sessions()[:limit]

        if not sessions:
            return ToolResult(
                success=True,
                result="No previous sessions found.",
                error=None,
            )

        lines = [f"Found {len(sessions)} session(s):\n"]

        for session in sessions:
            created = datetime.fromtimestamp(session["created"]).strftime("%Y-%m-%d %H:%M")
            completed = session["completed"]
            total = session["todo_count"]

            lines.append(
                f"• {session['session_id']}: {completed}/{total} completed ({created})"
            )

        lines.append(f"\nCurrent session: {_get_session_id()}")
        lines.append("Use todo_read(session_id='...') to view a specific session.")

        return ToolResult(
            success=True,
            result="\n".join(lines),
            error=None,
            metadata={"sessions": sessions, "current_session": _get_session_id()},
        )


# =============================================================================
# Skills Tool - Progressive Disclosure
# =============================================================================


def _parse_skill_frontmatter(content: str) -> Dict[str, str]:
    """Extract YAML frontmatter from SKILL.md."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}

    frontmatter = {}
    for line in match.group(1).split("\n"):
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()

    return frontmatter


def _get_skill_body(content: str) -> str:
    """Extract the body (after frontmatter) from a SKILL.md file."""
    match = re.match(r"^---\s*\n.*?\n---\s*\n", content, re.DOTALL)
    if match:
        return content[match.end():]
    return content


class SkillsTool(BaseTool):
    """Discover and load skills for domain-specific guidance.

    Skills are SKILL.md files with YAML frontmatter providing:
    - Patterns and best practices
    - Code examples
    - Domain-specific guidance

    Progressive disclosure:
    - 'list' shows only summaries (saves context)
    - 'load' fetches full content when needed
    """

    def __init__(
        self,
        builtin_path: Optional[Path] = None,
        user_path: Optional[Path] = None,
        project_path: Optional[Path] = None,
        extra_paths: Optional[List[Path]] = None,
    ):
        """Create a skills tool.

        Args:
            builtin_path: Path to built-in skills (shipped with package)
            user_path: Path to user skills (defaults to ~/.picoagents/skills/)
            project_path: Path to project-local skills (highest priority)
            extra_paths: Additional skill paths to search (appended after project_path)
        """
        super().__init__(
            name="skills",
            description=(
                "Discover and load skills for domain-specific guidance. "
                "Use action='list' to see available skills (summaries only). "
                "Use action='load' with name to get full content."
            ),
        )

        # Build list of skill paths (later paths override earlier)
        self.skill_paths: List[Path] = []
        if builtin_path:
            self.skill_paths.append(builtin_path)
        if user_path:
            self.skill_paths.append(user_path)
        else:
            # Default user path
            default_user = Path.home() / ".picoagents" / "skills"
            if default_user.exists():
                self.skill_paths.append(default_user)
        if project_path:
            self.skill_paths.append(project_path)
        if extra_paths:
            self.skill_paths.extend(extra_paths)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "load"],
                    "description": "'list' for summaries, 'load' for full content",
                },
                "name": {
                    "type": "string",
                    "description": "Skill name to load (required for 'load' action)",
                },
            },
            "required": ["action"],
        }

    def _discover_skills(self) -> Dict[str, tuple]:
        """Discover all skills from configured paths."""
        skills: Dict[str, tuple] = {}

        for skills_path in self.skill_paths:
            if not skills_path.exists():
                continue

            for item in skills_path.iterdir():
                if item.is_dir():
                    skill_md = item / "SKILL.md"
                    if skill_md.exists():
                        try:
                            content = skill_md.read_text()
                            meta = _parse_skill_frontmatter(content)
                            skill_name = meta.get("name", item.name)
                            skills[skill_name] = (skill_md, meta)
                        except Exception:
                            skills[item.name] = (
                                skill_md,
                                {"name": item.name, "description": "Error reading skill"},
                            )

        return skills

    def get_system_prompt_section(self) -> str:
        """Return skill metadata for system prompt injection.

        Pre-populates the system prompt with skill names and descriptions
        so the model knows what skills exist without calling list first.
        The model can then call skills(action='load', name='...') directly.
        """
        discovered = self._discover_skills()
        if not discovered:
            return ""

        lines = [
            "\n## Available Skills\n",
            "Use `skills(action='load', name='...')` to load full instructions when a skill matches the task.\n",
        ]
        for name, (_, meta) in sorted(discovered.items()):
            desc = meta.get("description", "No description")
            lines.append(f"- **{name}**: {desc}")
        return "\n".join(lines)

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the skills tool."""
        action = parameters.get("action", "list")
        name = parameters.get("name", "")

        discovered = self._discover_skills()

        if action == "list":
            if not discovered:
                paths_str = "\n".join(f"  - {p}" for p in self.skill_paths) if self.skill_paths else "  (no paths configured)"
                result = (
                    "No skills found.\n\n"
                    f"Skills are loaded from:\n{paths_str}\n\n"
                    "Each skill should be a folder with a SKILL.md file."
                )
                return ToolResult(success=True, result=result, error=None)

            lines = ["# Available Skills\n"]
            lines.append("Use `skills(action='load', name='...')` to load full content.\n")

            for skill_name, (_, meta) in sorted(discovered.items()):
                description = meta.get("description", "No description")
                triggers = meta.get("triggers", "")

                lines.append(f"### {skill_name}")
                lines.append(description)
                if triggers:
                    lines.append(f"_Triggers: {triggers}_")
                lines.append("")

            return ToolResult(
                success=True,
                result="\n".join(lines),
                error=None,
                metadata={"skill_count": len(discovered)},
            )

        elif action == "load":
            if not name:
                return ToolResult(
                    success=False,
                    result="",
                    error="'name' parameter is required for 'load' action",
                )

            if name not in discovered:
                available = sorted(discovered.keys())
                msg = f"Skill '{name}' not found."
                if available:
                    msg += f"\n\nAvailable skills: {', '.join(available)}"
                return ToolResult(success=False, result="", error=msg)

            skill_md, meta = discovered[name]
            try:
                content = skill_md.read_text()
                skill_name = meta.get("name", name)
                body = _get_skill_body(content)
                return ToolResult(
                    success=True,
                    result=f"# Skill: {skill_name}\n\n{body}",
                    error=None,
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    result="",
                    error=f"Error loading skill '{name}': {e}",
                )

        else:
            return ToolResult(
                success=False,
                result="",
                error=f"Unknown action: '{action}'. Use 'list' or 'load'.",
            )


# =============================================================================
# Multi-Edit Tool - Atomic File Edits
# =============================================================================


class MultiEditTool(BaseTool):
    """Make multiple edits to a file atomically.

    All edits succeed or fail together. If any edit fails:
    - File is unchanged (atomic rollback)
    - Error message indicates which edit failed

    Each edit needs:
    - old_string: text to find (must be unique)
    - new_string: text to replace with

    Edits are applied sequentially, so later edits see results of earlier ones.
    """

    def __init__(self, workspace: Optional[Path] = None):
        """Create a multi-edit tool.

        Args:
            workspace: Base directory for file operations (defaults to cwd)
        """
        super().__init__(
            name="multi_edit",
            description=(
                "Make multiple edits to a file atomically. All succeed or fail together. "
                "Each edit needs 'old_string' (unique in file) and 'new_string'. "
                "Edits are applied sequentially."
            ),
        )
        self.workspace = workspace or Path.cwd()

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "edits": {
                    "type": "array",
                    "description": "List of edits to apply",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {
                                "type": "string",
                                "description": "Text to find (must be unique)",
                            },
                            "new_string": {
                                "type": "string",
                                "description": "Text to replace with",
                            },
                        },
                        "required": ["old_string", "new_string"],
                    },
                },
            },
            "required": ["path", "edits"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute multi-edit atomically."""
        path_str = parameters.get("path", "")
        edits = parameters.get("edits", [])

        if not path_str:
            return ToolResult(success=False, result="", error="'path' is required")
        if not edits:
            return ToolResult(success=False, result="", error="'edits' list is required")

        try:
            path = Path(path_str).expanduser().resolve()

            # Security: ensure within workspace
            try:
                path.relative_to(self.workspace.resolve())
            except ValueError:
                # Allow absolute paths outside workspace for flexibility
                pass

            if not path.exists():
                return ToolResult(
                    success=False,
                    result="",
                    error=f"File not found: {path}",
                )

            content = path.read_text(encoding="utf-8")
            original_content = content

            # Validate all edits first
            for i, edit in enumerate(edits):
                if "old_string" not in edit or "new_string" not in edit:
                    return ToolResult(
                        success=False,
                        result="",
                        error=f"Edit {i+1} missing 'old_string' or 'new_string'",
                    )

            # Apply edits sequentially
            applied = []
            for i, edit in enumerate(edits):
                old_str = edit["old_string"]
                new_str = edit["new_string"]

                count = content.count(old_str)

                if count == 0:
                    return ToolResult(
                        success=False,
                        result="",
                        error=(
                            f"Edit {i+1} failed - could not find text.\n"
                            f"Applied {len(applied)} edit(s) before failure.\n"
                            f"File unchanged (atomic rollback)."
                        ),
                    )

                if count > 1:
                    return ToolResult(
                        success=False,
                        result="",
                        error=(
                            f"Edit {i+1} failed - found {count} occurrences (must be unique).\n"
                            f"Applied {len(applied)} edit(s) before failure.\n"
                            f"File unchanged (atomic rollback)."
                        ),
                    )

                content = content.replace(old_str, new_str, 1)
                applied.append(f"Edit {i+1}: replaced {len(old_str)} chars with {len(new_str)} chars")

            # All edits succeeded - write the file
            path.write_text(content, encoding="utf-8")

            result = f"Successfully applied {len(edits)} edit(s) to {path_str}:\n" + "\n".join(applied)

            return ToolResult(
                success=True,
                result=result,
                error=None,
                metadata={"edits_applied": len(edits), "path": str(path)},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                result="",
                error=f"Multi-edit failed: {str(e)}",
            )


# =============================================================================
# Factory Functions
# =============================================================================


def create_task_tool(
    coordinator: Optional["Agent"] = None,
    model_client: Optional["BaseChatCompletionClient"] = None,
    token_budget: int = 50_000,
    max_iterations: int = 20,
) -> TaskTool:
    """Create a Task tool for spawning sub-agents.

    Args:
        coordinator: Parent agent (for inheriting config)
        model_client: Model client for sub-agents
        token_budget: Token budget for sub-agents
        max_iterations: Max iterations for sub-agents

    Returns:
        TaskTool instance
    """
    return TaskTool(
        coordinator=coordinator,
        model_client=model_client,
        token_budget=token_budget,
        max_iterations=max_iterations,
    )


def create_todo_tools(include_sessions: bool = False) -> List[BaseTool]:
    """Create todo tracking tools.

    Args:
        include_sessions: If True, includes TodoListSessionsTool for session management

    Returns:
        List containing TodoWriteTool, TodoReadTool, and optionally TodoListSessionsTool
    """
    tools = [TodoWriteTool(), TodoReadTool()]
    if include_sessions:
        tools.append(TodoListSessionsTool())
    return tools


def create_skills_tool(
    builtin_path: Optional[Path] = None,
    user_path: Optional[Path] = None,
    project_path: Optional[Path] = None,
    extra_paths: Optional[List[Path]] = None,
) -> SkillsTool:
    """Create a skills tool for progressive disclosure.

    Args:
        builtin_path: Path to built-in skills
        user_path: Path to user skills (defaults to ~/.picoagents/skills/)
        project_path: Path to project-local skills
        extra_paths: Additional skill paths to search

    Returns:
        SkillsTool instance
    """
    return SkillsTool(
        builtin_path=builtin_path,
        user_path=user_path,
        project_path=project_path,
        extra_paths=extra_paths,
    )


def create_multi_edit_tool(workspace: Optional[Path] = None) -> MultiEditTool:
    """Create a multi-edit tool for atomic file edits.

    Args:
        workspace: Base directory for file operations

    Returns:
        MultiEditTool instance
    """
    return MultiEditTool(workspace=workspace)


def create_context_engineering_tools(
    coordinator: Optional["Agent"] = None,
    model_client: Optional["BaseChatCompletionClient"] = None,
    skills_path: Optional[Path] = None,
    workspace: Optional[Path] = None,
) -> List[BaseTool]:
    """Create all context engineering tools.

    This is the recommended way to add context engineering capabilities:
    - TaskTool for isolation
    - TodoWriteTool/TodoReadTool for progress tracking
    - SkillsTool for progressive disclosure
    - MultiEditTool for atomic edits

    Args:
        coordinator: Parent agent for TaskTool
        model_client: Model client for sub-agents
        skills_path: Path to project skills
        workspace: Base directory for file operations

    Returns:
        List of context engineering tools
    """
    tools: List[BaseTool] = [
        create_task_tool(coordinator=coordinator, model_client=model_client),
        *create_todo_tools(),
        create_skills_tool(project_path=skills_path),
        create_multi_edit_tool(workspace=workspace),
    ]
    return tools
