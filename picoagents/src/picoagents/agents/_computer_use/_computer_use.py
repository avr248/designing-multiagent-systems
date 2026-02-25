"""
ComputerUseAgent - A tool-based agent for computer interface automation.

This module provides the ComputerUseAgent class that inherits from the base Agent
and adds computer interface automation capabilities through specialized tools.
"""

from collections.abc import AsyncGenerator
from typing import List, Optional, Union

from ..._cancellation_token import CancellationToken
from ...context import AgentContext
from ...llm import BaseChatCompletionClient
from ...messages import (
    AssistantMessage,
    Message,
    MultiModalMessage,
    ToolMessage,
    UserMessage,
)
from ...types import AgentEvent, AgentResponse, Usage
from .._agent import Agent
from ._interface_clients import BaseInterfaceClient
from ._playwright_tools import create_playwright_tools


class ComputerUseAgent(Agent):
    """
    Computer interface automation agent using tool calling.

    Inherits from the base Agent class and adds specialized tools for
    web interface automation. Uses the proven tool calling patterns
    from the base implementation.
    """

    def __init__(
        self,
        interface_client: BaseInterfaceClient,
        model_client: BaseChatCompletionClient,
        name: str = "computer_navigator",
        description: str = "Agent that uses tools to interact with web interfaces",
        use_screenshots: bool = True,
        max_actions: int = 20,
        **kwargs,
    ):
        """
        Initialize ComputerUseAgent.

        Args:
            interface_client: Backend for interface automation
            model_client: LLM client for decision making
            name: Agent name
            description: Agent description
            use_screenshots: Whether to provide screenshots to the LLM and stream them to UI
            max_actions: Maximum actions per task
            **kwargs: Additional Agent configuration
        """
        # Create Playwright tools
        playwright_tools = create_playwright_tools(interface_client)

        # Set up instructions if not provided
        if "instructions" not in kwargs:
            kwargs["instructions"] = self._default_instructions(playwright_tools)

        # Pass tools to the base Agent
        super().__init__(
            name=name,
            description=description,
            model_client=model_client,
            tools=playwright_tools,  # type: ignore
            max_iterations=max_actions,
            **kwargs,
        )

        self.interface_client = interface_client
        self.use_screenshots = use_screenshots
        self.is_initialized = False

    def _default_instructions(self, tools) -> str:
        """
        Generate default instructions dynamically based on available tools.

        Args:
            tools: List of tools available to this agent

        Returns:
            Instructions string with tool descriptions
        """
        # Build tool list directly from tool objects - no hardcoding!
        tool_list = []
        for tool in tools:
            tool_list.append(f"- {tool.name}: {tool.description}")

        # Join all tool descriptions
        tools_text = "\n".join(tool_list)

        return f"""You are an efficient computer use agent focused on task completion.

COMPLETION-FIRST MINDSET:
- After each action, ask yourself: "Do I now have enough information to complete the task?"
- If YES, immediately provide your answer and stop
- If NO, take the next minimal action needed
- Don't perform unnecessary actions once you have the answer

Your process:
1. Understand what specific information you need for the task
2. Take the minimum actions needed to get that information
3. As soon as you have it, provide your answer immediately
4. Only continue if you're missing critical information

Available tools:
{tools_text}

EFFICIENCY EXAMPLES:
Task: "Find the latest blog post title"
✅ navigate(blog-url) → observe_page() → provide answer
❌ navigate → observe → scroll → click → scroll → eventually stop

Task: "Get company CEO name from about page"
✅ navigate(about-url) → observe_page() → provide answer "John Smith"
❌ navigate → observe → click links → scroll → eventually find it

Be decisive: if you can see the answer, provide it immediately!"""

    async def run_stream(
        self,
        task: Optional[Union[str, UserMessage, List[Message]]] = None,
        context: Optional[AgentContext] = None,
        cancellation_token: Optional[CancellationToken] = None,
        verbose: bool = False,
        stream_tokens: bool = False,
    ) -> AsyncGenerator[Union[Message, AgentEvent, AgentResponse], None]:
        """
        Execute task with screenshot support.

        Inherits tool calling from base Agent and adds computer interface features.

        Args:
            task: The task or query for the agent to address
            context: Optional context (passed to parent Agent.run_stream)
            cancellation_token: Optional token for cancelling execution
            verbose: Enable detailed event logging
            stream_tokens: Enable token-level streaming from LLM
        """
        # Initialize interface
        if not self.is_initialized:
            await self.interface_client.initialize()
            self.is_initialized = True

        # Resolve the working context so we can inject screenshots into it.
        # Parent's run_stream() uses this same object as its working_context
        # when an explicit context is passed, keeping everything in sync.
        working_context = context if context else self.context.model_copy(deep=True)

        # Capture initial screenshot for UI display
        initial_state = await self.interface_client.get_state("hybrid")
        if initial_state.screenshot:
            yield MultiModalMessage(
                content=f"Initial page - URL: {initial_state.url or 'N/A'}",
                source=self.name,
                role="user",
                mime_type="image/png",
                data=initial_state.screenshot,
                media_url=None,
            )

        # Use base Agent's run_stream but intercept tool events for screenshots and task completion
        async for item in super().run_stream(
            task=task,
            context=working_context,
            cancellation_token=cancellation_token,
            verbose=verbose,
            stream_tokens=stream_tokens,
        ):
            yield item

            # Handle observe_page results - add screenshot to context if use_screenshots is enabled
            if isinstance(item, ToolMessage) and item.tool_name == "observe_page":
                if self.use_screenshots:
                    try:
                        # Get current state with screenshot
                        state = await self.interface_client.get_state("hybrid")
                        if state.screenshot:
                            screenshot_msg = MultiModalMessage(
                                content=f"Page observation",
                                source=self.name,
                                role="user",
                                mime_type="image/png",
                                data=state.screenshot,
                                media_url=None,
                            )
                            working_context.add_message(screenshot_msg)
                            # Also yield for UI display
                            yield screenshot_msg
                    except Exception:
                        # Don't fail if screenshot capture fails
                        pass

            # Capture screenshots after tool calls for UI display (except observe_page)
            if isinstance(item, ToolMessage) and item.tool_name != "observe_page":
                try:
                    new_state = await self.interface_client.get_state("hybrid")
                    if new_state.screenshot:
                        yield MultiModalMessage(
                            content=f"After {item.tool_name} - URL: {new_state.url or 'N/A'}",
                            source=self.name,
                            role="user",
                            mime_type="image/png",
                            data=new_state.screenshot,
                            media_url=None,
                        )
                except Exception:
                    # Don't fail the whole execution if screenshot capture fails
                    pass

    async def reset(self) -> None:
        """Reset the agent state."""
        await super().reset()
        if hasattr(self, "is_initialized") and self.is_initialized:
            await self.interface_client.close()
            self.is_initialized = False

    async def close(self) -> None:
        """Close the interface client."""
        if hasattr(self, "is_initialized") and self.is_initialized:
            await self.interface_client.close()
            self.is_initialized = False

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager with browser cleanup."""
        await self.close()
        return await super().__aexit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        """Cleanup browser on deletion as safety net."""
        if hasattr(self, "is_initialized") and self.is_initialized:
            try:
                # Try to close synchronously
                import asyncio

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, schedule cleanup
                        loop.create_task(self.close())
                    else:
                        # If no loop, run cleanup
                        asyncio.run(self.close())
                except RuntimeError:
                    # If we can't get a loop, just pass
                    pass
            except Exception:
                # Silently fail - destructor shouldn't raise
                pass
