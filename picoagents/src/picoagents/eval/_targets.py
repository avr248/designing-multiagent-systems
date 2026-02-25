"""
Evaluation targets - what we run tasks against.

This module provides concrete Target implementations for running tasks
against different systems: PicoAgents, direct model calls, orchestrators,
Claude Code SDK, and arbitrary callables.
"""

import time
from typing import Any, Dict, List, Optional

from .._cancellation_token import CancellationToken
from ..agents import BaseAgent
from ..llm import BaseChatCompletionClient
from ..messages import SystemMessage, UserMessage
from ..orchestration import BaseOrchestrator
from ..types import EvalScore, RunTrajectory, Task, Usage
from ._base import Target
from ._config import AgentConfig


class AgentEvalTarget(Target):
    """Target that wraps a picoagents Agent.

    Safe for concurrent use: Agent.run() and run_stream() use local
    working_context variables internally, so parallel task execution
    (parallel_tasks=True) does not cause shared-state races.
    """

    def __init__(self, agent: BaseAgent, name: Optional[str] = None):
        super().__init__(name or getattr(agent, "name", "Agent"))
        self.agent = agent

    async def run(
        self, task: Task, cancellation_token: Optional[CancellationToken] = None
    ) -> RunTrajectory:
        start_time = time.time()

        try:
            response = await self.agent.run(
                task.input, cancellation_token=cancellation_token
            )

            end_time = time.time()

            return RunTrajectory(
                task=task,
                messages=response.messages,
                success=True,
                error=None,
                usage=response.usage,
                metadata={
                    "target_type": "agent",
                    "target_name": self.name,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )

        except Exception as e:
            end_time = time.time()

            return RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(
                    duration_ms=int((end_time - start_time) * 1000),
                    llm_calls=0,
                    tokens_input=0,
                    tokens_output=0,
                ),
                metadata={
                    "target_type": "agent",
                    "target_name": self.name,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )


class ModelEvalTarget(Target):
    """Target for direct LLM model calls."""

    def __init__(
        self,
        client: BaseChatCompletionClient,
        system_message: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name or getattr(client, "model", "Model"))
        self.client = client
        self.system_message = system_message

    async def run(
        self, task: Task, cancellation_token: Optional[CancellationToken] = None
    ) -> RunTrajectory:
        start_time = time.time()

        try:
            messages = []
            if self.system_message:
                messages.append(
                    SystemMessage(content=self.system_message, source="system")
                )
            messages.append(UserMessage(content=task.input, source="user"))

            result = await self.client.create(messages)

            end_time = time.time()

            response_messages = messages + [result.message]

            return RunTrajectory(
                task=task,
                messages=response_messages,
                success=True,
                error=None,
                usage=result.usage,
                metadata={
                    "target_type": "model",
                    "target_name": self.name,
                    "model": result.model,
                    "finish_reason": result.finish_reason,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )

        except Exception as e:
            end_time = time.time()

            return RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(
                    duration_ms=int((end_time - start_time) * 1000),
                    llm_calls=0,
                    tokens_input=0,
                    tokens_output=0,
                ),
                metadata={
                    "target_type": "model",
                    "target_name": self.name,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )


class OrchestratorEvalTarget(Target):
    """Target for picoagents orchestrators."""

    def __init__(self, orchestrator: BaseOrchestrator, name: Optional[str] = None):
        super().__init__(name or f"{orchestrator.__class__.__name__}")
        self.orchestrator = orchestrator

    async def run(
        self, task: Task, cancellation_token: Optional[CancellationToken] = None
    ) -> RunTrajectory:
        start_time = time.time()

        try:
            response = await self.orchestrator.run(
                task.input, cancellation_token=cancellation_token
            )

            end_time = time.time()

            return RunTrajectory(
                task=task,
                messages=response.messages,
                success=True,
                error=None,
                usage=response.usage,
                metadata={
                    "target_type": "orchestrator",
                    "target_name": self.name,
                    "pattern": response.pattern_metadata.get("pattern", "unknown"),
                    "iterations": response.pattern_metadata.get(
                        "iterations_completed", 0
                    ),
                    "stop_reason": response.stop_message.source,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )

        except Exception as e:
            end_time = time.time()

            return RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(
                    duration_ms=int((end_time - start_time) * 1000),
                    llm_calls=0,
                    tokens_input=0,
                    tokens_output=0,
                ),
                metadata={
                    "target_type": "orchestrator",
                    "target_name": self.name,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )


class PicoAgentTarget(Target):
    """Target that creates an agent from an AgentConfig and runs tasks.

    Uses run_stream to capture the full message and event trace.
    """

    def __init__(
        self,
        config: AgentConfig,
        middlewares: Optional[List] = None,
    ):
        super().__init__(config.name)
        self.config = config
        self.middlewares = middlewares or []

    def _get_agent(self, extra_middlewares: Optional[List] = None):
        """Create agent with combined middleware."""
        all_middlewares = self.middlewares + (extra_middlewares or [])
        return self.config.to_agent(middlewares=all_middlewares)

    async def run(
        self,
        task: Task,
        cancellation_token: Optional[CancellationToken] = None,
        *,
        middlewares: Optional[List] = None,
    ) -> RunTrajectory:
        """Execute task with PicoAgents using run_stream to capture full trace.

        Args:
            task: Task to run
            cancellation_token: Optional cancellation
            middlewares: Additional middleware (e.g., RunMiddleware injected by runner)

        Returns:
            RunTrajectory with execution data including full message/event history
        """
        from ..messages import AssistantMessage, Message, ToolMessage
        from ..types import AgentEvent, AgentResponse

        agent = self._get_agent(middlewares)

        try:
            all_messages: List[Any] = []
            all_events: List[Any] = []
            response = None
            output = ""

            async for item in agent.run_stream(
                task.input,
                cancellation_token=cancellation_token,
                verbose=True,
            ):
                if isinstance(item, AgentResponse):
                    response = item
                elif isinstance(item, Message):
                    all_messages.append(item)
                    if isinstance(item, AssistantMessage) and item.content:
                        output = item.content
                elif isinstance(item, AgentEvent):
                    all_events.append(item)

            if response is None:
                return RunTrajectory(
                    task=task,
                    messages=all_messages,
                    success=False,
                    error="No response from agent",
                    usage=Usage(duration_ms=0, llm_calls=0, tokens_input=0, tokens_output=0),
                    metadata={"exception_type": "NoResponse", "events": all_events},
                )

            # Get messages from context if available (more complete)
            context_messages = list(response.context.messages) if response.context else []

            # Build events metadata
            metadata: Dict[str, Any] = {
                "finish_reason": response.finish_reason,
                "tool_calls": response.usage.tool_calls,
            }
            if all_events:
                metadata["events"] = [
                    {
                        "type": type(e).__name__,
                        "source": getattr(e, "source", None),
                        **{k: v for k, v in vars(e).items() if k != "source" and not k.startswith("_")}
                    }
                    for e in all_events
                ]
                metadata["event_count"] = len(all_events)

            return RunTrajectory(
                task=task,
                messages=context_messages if context_messages else all_messages,
                success=response.finish_reason == "stop",
                error=None if response.finish_reason == "stop" else response.finish_reason,
                usage=Usage(
                    duration_ms=response.usage.duration_ms,
                    llm_calls=response.usage.llm_calls,
                    tokens_input=response.usage.tokens_input,
                    tokens_output=response.usage.tokens_output,
                    tool_calls=response.usage.tool_calls,
                ),
                metadata=metadata,
            )

        except Exception as e:
            return RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(duration_ms=0, llm_calls=0, tokens_input=0, tokens_output=0),
                metadata={"exception_type": type(e).__name__},
            )


class ClaudeCodeTarget(Target):
    """Target that runs tasks with Claude Code SDK.

    Requires `claude-code-sdk` package to be installed.
    """

    def __init__(
        self,
        name: str = "claude_code",
        max_turns: int = 30,
        allowed_tools: Optional[List[str]] = None,
    ):
        super().__init__(name)
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools or ["Read", "Bash", "Glob", "Grep"]

    async def run(
        self, task: Task, cancellation_token: Optional[CancellationToken] = None
    ) -> RunTrajectory:
        try:
            from claude_code_sdk import (
                AssistantMessage as CCAssistantMessage,
                ClaudeCodeOptions,
                ResultMessage,
                TextBlock,
                query,
            )
        except ImportError:
            return RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error="claude-code-sdk not installed. Install with: pip install claude-code-sdk",
                usage=Usage(duration_ms=0, llm_calls=0, tokens_input=0, tokens_output=0),
            )

        options = ClaudeCodeOptions(
            allowed_tools=self.allowed_tools,
            max_turns=self.max_turns,
        )

        response_text = ""
        iterations = 0
        input_tokens = 0
        output_tokens = 0
        duration_ms = 0
        success = False
        error = None

        try:
            async for message in query(prompt=task.input, options=options):
                if isinstance(message, CCAssistantMessage):
                    iterations += 1
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text

                elif isinstance(message, ResultMessage):
                    success = not message.is_error
                    duration_ms = message.duration_ms
                    if message.usage:
                        input_tokens = message.usage.get("input_tokens", 0)
                        output_tokens = message.usage.get("output_tokens", 0)
                    if message.is_error:
                        error = "Claude Code returned error"

        except Exception as e:
            error = str(e)

        # Build minimal message history
        from ..messages import AssistantMessage, UserMessage
        messages = [
            UserMessage(content=task.input, source="user"),
        ]
        if response_text:
            messages.append(AssistantMessage(content=response_text, source="assistant"))

        return RunTrajectory(
            task=task,
            messages=messages,
            success=success,
            error=error,
            usage=Usage(
                duration_ms=duration_ms,
                llm_calls=iterations,
                tokens_input=input_tokens,
                tokens_output=output_tokens,
            ),
        )


class CallableTarget(Target):
    """Wrap any async callable as a target.

    Useful for custom agent implementations or quick testing.
    The callable receives a Task and returns a RunTrajectory.
    """

    def __init__(self, name: str, func):
        super().__init__(name)
        self.func = func

    async def run(
        self, task: Task, cancellation_token: Optional[CancellationToken] = None
    ) -> RunTrajectory:
        """Execute the wrapped callable."""
        return await self.func(task)
