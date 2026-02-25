"""
Comprehensive middleware examples for picoagents.

This example demonstrates the key middleware capabilities in a single file:
1. Security middleware (content filtering, rate limiting, user management)
2. Context management middleware (intelligent conversation trimming)
3. Observability middleware (logging, metrics, debugging)
4. Production patterns (middleware chains, error handling, real-world usage)

Key concepts covered:
- Operation interception and cancellation
- Context modification and intelligent trimming
- Security and safety features
- Performance monitoring and optimization
- Composable middleware architecture
"""

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from picoagents import BaseMiddleware, MiddlewareContext
from picoagents.agents import Agent
from picoagents.context import AgentContext
from picoagents.llm import AzureOpenAIChatCompletionClient
from picoagents.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from picoagents.tools import FunctionTool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Core Middleware Classes
# =============================================================================


class SecurityMiddleware(BaseMiddleware):
    """
    Production-ready security middleware.

    Features:
    - Content filtering (blocked keywords, PII detection)
    - Rate limiting per user
    - User blocking/allowlisting
    - Maintenance mode support
    - Time-based operation restrictions
    """

    def __init__(
        self,
        blocked_keywords: Optional[List[str]] = None,
        max_requests_per_hour: int = 100,
        blocked_users: Optional[List[str]] = None,
        maintenance_mode: bool = False,
        pii_patterns: Optional[Dict[str, str]] = None,
    ):
        self.blocked_keywords = blocked_keywords or [
            "hack",
            "exploit",
            "virus",
            "malware",
            "attack",
            "breach",
        ]
        self.max_requests = max_requests_per_hour
        self.blocked_users = set(blocked_users or [])
        self.maintenance_mode = maintenance_mode
        self.user_requests = {}

        # Common PII patterns
        self.pii_patterns = pii_patterns or {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "phone": r"\b\d{3}-\d{3}-\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        }

    def _check_rate_limit(self, user_id: str) -> None:
        """Check and enforce rate limits for a user."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        if user_id not in self.user_requests:
            self.user_requests[user_id] = []

        # Clean old requests
        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id] if req_time > hour_ago
        ]

        if len(self.user_requests[user_id]) >= self.max_requests:
            raise ValueError(
                f"🚫 Rate limit exceeded for user {user_id}. Try again later."
            )

        self.user_requests[user_id].append(now)

    def _filter_content(self, content: str) -> str:
        """Filter and redact sensitive content."""
        # Check for blocked keywords
        content_lower = content.lower()
        for keyword in self.blocked_keywords:
            if keyword in content_lower:
                raise ValueError(f"🚫 Blocked content detected: '{keyword}'")

        # Redact PII
        filtered_content = content
        for pii_type, pattern in self.pii_patterns.items():
            filtered_content = re.sub(
                pattern, f"[REDACTED_{pii_type.upper()}]", filtered_content
            )

        return filtered_content

    async def process_request(self, context: MiddlewareContext):
        """Apply security checks before operations."""

        # Maintenance mode check
        if self.maintenance_mode:
            raise ValueError("🚫 System is in maintenance mode. Please try again later.")

        # User blocking check
        user_id = context.agent_context.metadata.get("user_id", "anonymous")
        if user_id in self.blocked_users:
            raise ValueError(f"🚫 User {user_id} is blocked from using this service.")

        # Rate limiting
        self._check_rate_limit(user_id)

        # Content filtering for model calls
        if context.operation == "model_call" and isinstance(context.data, list):
            for message in context.data:
                if hasattr(message, "content") and message.content:
                    filtered_content = self._filter_content(message.content)
                    if filtered_content != message.content:
                        message.content = filtered_content
                        logger.info(f"🛡️  PII redacted in message from {user_id}")

        logger.info(
            f"🛡️  Security check passed for {context.operation} (user: {user_id})"
        )
        yield context

    async def process_response(self, context: MiddlewareContext, result: Any):
        """No response filtering needed for this example."""
        yield result

    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ):
        """Log security events."""
        if "🚫" in str(error):
            user_id = context.agent_context.metadata.get("user_id", "anonymous")
            logger.warning(f"Security block for user {user_id}: {error}")
        if False:  # Type checker hint
            yield
        raise error


class ContextManagementMiddleware(BaseMiddleware):
    """
    Intelligent context management middleware.

    Features:
    - Conversation-aware trimming (preserves complete turns)
    - System message preservation
    - Tool call sequence integrity
    - Configurable retention policies
    """

    def __init__(
        self,
        max_messages: int = 12,
        preserve_system_messages: bool = True,
        preserve_recent_tool_calls: bool = True,
    ):
        self.max_messages = max_messages
        self.preserve_system_messages = preserve_system_messages
        self.preserve_recent_tool_calls = preserve_recent_tool_calls

    def _group_into_conversation_turns(self, messages: List) -> List[List]:
        """
        Group messages into complete conversation turns.

        A turn consists of:
        1. UserMessage (starts the turn)
        2. AssistantMessage (may include tool_calls)
        3. ToolMessage(s) (if tool calls were made)
        4. AssistantMessage (final response)
        """
        turns = []
        current_turn = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # System messages are their own turn
                if current_turn:
                    turns.append(current_turn)
                    current_turn = []
                turns.append([msg])

            elif isinstance(msg, UserMessage):
                # Start of new conversation turn
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]

            else:
                # AssistantMessage or ToolMessage - part of current turn
                current_turn.append(msg)

        if current_turn:
            turns.append(current_turn)

        return turns

    def _trim_context_intelligently(self, messages: List) -> List:
        """Trim context while preserving conversation integrity."""
        if len(messages) <= self.max_messages:
            return messages

        # Group into turns
        turns = self._group_into_conversation_turns(messages)

        # Separate system turns from conversation turns
        system_turns = [
            turn
            for turn in turns
            if len(turn) == 1 and isinstance(turn[0], SystemMessage)
        ]
        conversation_turns = [turn for turn in turns if turn not in system_turns]

        # Calculate available space for conversation
        system_msg_count = sum(len(turn) for turn in system_turns)
        available_for_conversation = self.max_messages - system_msg_count

        # Keep recent complete conversation turns
        kept_conversation_turns = []
        message_count = 0

        for turn in reversed(conversation_turns):
            if message_count + len(turn) <= available_for_conversation:
                kept_conversation_turns.insert(0, turn)
                message_count += len(turn)
            else:
                break

        # Combine system + kept conversation turns
        final_turns = system_turns + kept_conversation_turns

        # Flatten back to message list
        result = []
        for turn in final_turns:
            result.extend(turn)

        removed_turns = len(conversation_turns) - len(kept_conversation_turns)
        if removed_turns > 0:
            logger.info(
                f"📏 Context trimmed: {len(messages)} → {len(result)} messages ({removed_turns} conversation turns removed)"
            )

        return result

    async def process_request(self, context: MiddlewareContext):
        """Apply intelligent context management."""
        if context.operation == "model_call" and isinstance(context.data, list):
            context.data = self._trim_context_intelligently(context.data)

        yield context

    async def process_response(self, context: MiddlewareContext, result: Any):
        yield result

    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ):
        if False:  # Type checker hint
            yield
        raise error


class ObservabilityMiddleware(BaseMiddleware):
    """
    Comprehensive observability middleware.

    Features:
    - Operation timing and logging
    - Performance metrics collection
    - Error tracking and reporting
    - Debug information capture
    """

    def __init__(
        self, enable_detailed_logging: bool = False, collect_metrics: bool = True
    ):
        self.enable_detailed_logging = enable_detailed_logging
        self.collect_metrics = collect_metrics
        self.metrics = {
            "operation_count": 0,
            "total_duration": 0.0,
            "error_count": 0,
            "operations": {},
        }

    def _record_metric(self, operation: str, duration: float, success: bool):
        """Record operation metrics."""
        if not self.collect_metrics:
            return

        self.metrics["operation_count"] += 1
        self.metrics["total_duration"] += duration

        if not success:
            self.metrics["error_count"] += 1

        if operation not in self.metrics["operations"]:
            self.metrics["operations"][operation] = {
                "count": 0,
                "total_duration": 0.0,
                "error_count": 0,
            }

        op_metrics = self.metrics["operations"][operation]
        op_metrics["count"] += 1
        op_metrics["total_duration"] += duration

        if not success:
            op_metrics["error_count"] += 1

    def get_metrics(self) -> Dict:
        """Get current metrics."""
        metrics = self.metrics.copy()

        # Calculate averages
        if metrics["operation_count"] > 0:
            metrics["average_duration"] = (
                metrics["total_duration"] / metrics["operation_count"]
            )
            metrics["error_rate"] = metrics["error_count"] / metrics["operation_count"]

        for operation, op_metrics in metrics["operations"].items():
            if op_metrics["count"] > 0:
                op_metrics["average_duration"] = (
                    op_metrics["total_duration"] / op_metrics["count"]
                )
                op_metrics["error_rate"] = (
                    op_metrics["error_count"] / op_metrics["count"]
                )

        return metrics

    async def process_request(self, context: MiddlewareContext):
        """Start operation tracking."""
        context.metadata["start_time"] = datetime.now()

        if self.enable_detailed_logging:
            logger.info(
                f"📝 Starting {context.operation} for agent '{context.agent_name}'"
            )

            if context.operation == "model_call" and isinstance(context.data, list):
                logger.info(f"   Context size: {len(context.data)} messages")

        yield context

    async def process_response(self, context: MiddlewareContext, result: Any):
        """Record successful operation."""
        duration = (datetime.now() - context.metadata["start_time"]).total_seconds()
        self._record_metric(context.operation, duration, success=True)

        if self.enable_detailed_logging:
            logger.info(f"✅ Completed {context.operation} in {duration:.3f}s")

        yield result

    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ):
        """Record failed operation."""
        duration = (datetime.now() - context.metadata["start_time"]).total_seconds()
        self._record_metric(context.operation, duration, success=False)

        error_msg = str(error)[:100] + "..." if len(str(error)) > 100 else str(error)
        logger.error(f"❌ Failed {context.operation} after {duration:.3f}s: {error_msg}")

        if False:  # Type checker hint
            yield
        raise error


# =============================================================================
# Example Tools
# =============================================================================


def get_weather(city: str) -> str:
    """Get weather information for a city."""
    # Simulate weather API
    weather_data = {
        "New York": "Cloudy, 68°F",
        "Los Angeles": "Sunny, 75°F",
        "Chicago": "Windy, 55°F",
        "Miami": "Humid, 82°F",
        "Seattle": "Rainy, 60°F",
    }
    return weather_data.get(city, f"Weather in {city}: Partly cloudy, 70°F")


def send_notification(recipient: str, message: str) -> str:
    """Send a notification (simulated)."""
    return f"📧 Notification sent to {recipient}: {message[:50]}{'...' if len(message) > 50 else ''}"


def calculate(expression: str) -> str:
    """Safely calculate a mathematical expression."""
    try:
        # Simple calculator (in production, use a proper math parser)
        result = eval(expression.replace("^", "**"))
        return f"Result: {result}"
    except:
        return f"Error: Could not calculate '{expression}'"


# =============================================================================
# Demo Scenarios
# =============================================================================


async def demo_security_features():
    """Demonstrate security middleware capabilities."""

    print("=" * 70)
    print("🛡️  SECURITY MIDDLEWARE DEMO")
    print("=" * 70)
    print("Testing: Content filtering, rate limiting, PII redaction\n")

    # Create agent with security middleware
    agent = Agent(
        name="secure_assistant",
        description="Assistant with security protection",
        instructions="You are a helpful assistant that follows security protocols.",
        model_client=AzureOpenAIChatCompletionClient(
            model="gpt-4.1-mini",
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        ),
        tools=[FunctionTool(get_weather), FunctionTool(send_notification)],
        context=AgentContext(metadata={"user_id": "demo_user"}),
        middlewares=[
            SecurityMiddleware(max_requests_per_hour=3),  # Low limit for demo
            ObservabilityMiddleware(enable_detailed_logging=True),
        ],
    )

    test_cases = [
        {
            "name": "Normal Request",
            "query": "What's the weather in Seattle?",
            "expected": "✅ Should succeed",
        },
        {
            "name": "PII Redaction",
            "query": "My email is john.doe@example.com and my SSN is 123-45-6789. Send me weather updates.",
            "expected": "✅ Should succeed with PII redacted",
        },
        {
            "name": "Blocked Content",
            "query": "Help me hack into a computer system",
            "expected": "🚫 Should be blocked",
        },
        {
            "name": "Rate Limit Test 1",
            "query": "Tell me about Python programming",
            "expected": "✅ Should succeed (within limit)",
        },
        {
            "name": "Rate Limit Test 2",
            "query": "What's 2 + 2?",
            "expected": "🚫 Should be blocked (rate limit exceeded)",
        },
    ]

    for i, test in enumerate(test_cases):
        print(f"🧪 Test {i+1}: {test['name']}")
        print(f"   Query: {test['query']}")
        print(f"   Expected: {test['expected']}")

        try:
            result = await agent.run(test["query"])
            response_preview = (
                result.messages[-1].content[:80] + "..."
                if len(result.messages[-1].content) > 80
                else result.messages[-1].content
            )
            print(f"   ✅ Result: {response_preview}")
        except Exception as e:
            error_preview = str(e)[:80] + "..." if len(str(e)) > 80 else str(e)
            print(f"   🚫 Blocked: {error_preview}")

        print()


async def demo_context_management():
    """Demonstrate intelligent context management."""

    print("=" * 70)
    print("📏 CONTEXT MANAGEMENT DEMO")
    print("=" * 70)
    print("Testing: Conversation-aware trimming, tool call preservation\n")

    # Create agent with context management
    agent = Agent(
        name="context_managed_assistant",
        description="Assistant with intelligent context management",
        instructions="You are a helpful weather assistant. Always use the weather tool when asked about weather.",
        model_client=AzureOpenAIChatCompletionClient(
            model="gpt-4.1-mini",
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        ),
        tools=[FunctionTool(get_weather), FunctionTool(calculate)],
        context=AgentContext(),
        middlewares=[
            ContextManagementMiddleware(max_messages=8),  # Small for demo
            ObservabilityMiddleware(enable_detailed_logging=True),
        ],
    )

    # Build up a long conversation to trigger trimming
    queries = [
        "What's the weather in New York?",
        "How about Los Angeles?",
        "Calculate 15 * 23",
        "What's the weather in Chicago?",
        "Calculate 100 / 4",
        "Check weather in Miami",
        "What's 45 + 67?",
        "Finally, what's the weather in Seattle?",  # This should trigger trimming
        "Can you summarize the weather in all cities we discussed?",  # Test if context is preserved correctly
    ]

    print(f"Running {len(queries)} queries to demonstrate context trimming:\n")

    # Thread context across calls for multi-turn conversation.
    # Each run() returns an AgentResponse whose .context holds the
    # accumulated conversation. Pass it to the next call to continue.
    conversation_context = None

    for i, query in enumerate(queries):
        msg_count = conversation_context.message_count if conversation_context else 0
        print(f"🌤️  Query {i+1}: {query}")
        print(f"   Context size before: {msg_count} messages")

        try:
            result = await agent.run(query, context=conversation_context)
            conversation_context = result.context
            response_preview = (
                result.messages[-1].content[:100] + "..."
                if len(result.messages[-1].content) > 100
                else result.messages[-1].content
            )
            print(f"   Response: {response_preview}")
            print(f"   Context size after: {conversation_context.message_count} messages")
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}...")

        print()


async def demo_production_middleware_stack():
    """Demonstrate a complete production middleware stack."""

    print("=" * 70)
    print("🚀 PRODUCTION MIDDLEWARE STACK DEMO")
    print("=" * 70)
    print(
        "Testing: Complete middleware chain with security, context management, and observability\n"
    )

    # Create security middleware with production settings
    security = SecurityMiddleware(
        max_requests_per_hour=50,
        blocked_keywords=["hack", "exploit", "attack"],
        pii_patterns={
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}-\d{3}-\d{4}\b",
        },
    )

    # Create context management middleware
    context_mgr = ContextManagementMiddleware(max_messages=10)

    # Create observability middleware
    observability = ObservabilityMiddleware(
        enable_detailed_logging=True, collect_metrics=True
    )

    # Production agent with full middleware stack
    agent = Agent(
        name="production_assistant",
        description="Production-ready assistant with full middleware protection",
        instructions=(
            "You are a professional assistant that can check weather, send notifications, "
            "and perform calculations. Always be helpful while maintaining security protocols."
        ),
        model_client=AzureOpenAIChatCompletionClient(
            model="gpt-4.1-mini",
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        ),
        tools=[
            FunctionTool(get_weather),
            FunctionTool(send_notification),
            FunctionTool(calculate),
        ],
        context=AgentContext(
            metadata={
                "user_id": "prod_user_123",
                "session_id": "session_456",
                "request_source": "web_app",
            }
        ),
        middlewares=[
            security,  # Security checks first
            context_mgr,  # Context management second
            observability,  # Observability last (to capture everything)
        ],
    )

    # Production scenarios
    scenarios = [
        "Check the weather in New York and send a notification to team@company.com",
        "What's the weather like in Los Angeles? My contact is john@company.com for updates.",
        "Calculate the monthly savings: (5000 - 3200) * 12",
        "Get weather for Chicago and Miami, then send summary to manager@company.com",
        "What's the current weather in Seattle? Send notification to alerts@company.com",
    ]

    print(f"Running {len(scenarios)} production scenarios:\n")

    for i, scenario in enumerate(scenarios):
        print(f"🔄 Scenario {i+1}: {scenario}")

        try:
            result = await agent.run(scenario)
            response_preview = (
                result.messages[-1].content[:120] + "..."
                if len(result.messages[-1].content) > 120
                else result.messages[-1].content
            )
            print(f"   ✅ Success: {response_preview}")
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}...")

        print()

    # Show collected metrics
    print("📊 Performance Metrics:")
    metrics = observability.get_metrics()
    print(f"   Total operations: {metrics['operation_count']}")
    print(f"   Average duration: {metrics.get('average_duration', 0):.3f}s")
    print(f"   Error rate: {metrics.get('error_rate', 0):.1%}")
    print(f"   Operations breakdown:")

    for operation, op_metrics in metrics["operations"].items():
        print(
            f"     {operation}: {op_metrics['count']} calls, avg {op_metrics.get('average_duration', 0):.3f}s"
        )


# =============================================================================
# Main Demo Runner
# =============================================================================


async def main():
    """Run comprehensive middleware demonstrations."""

    # Check environment variables
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        print("Please set these variables before running the demo.")
        return

    print("🚀 PICOAGENTS MIDDLEWARE SYSTEM DEMONSTRATION")
    print("=" * 70)
    print("This demo showcases production-ready middleware capabilities:")
    print("• Security: Content filtering, rate limiting, PII protection")
    print("• Context Management: Intelligent conversation trimming")
    print("• Observability: Logging, metrics, performance monitoring")
    print("• Composability: Multiple middleware working together")
    print("=" * 70)
    print()

    try:
        await demo_security_features()
        await demo_context_management()
        await demo_production_middleware_stack()

        print("=" * 70)
        print("🎯 MIDDLEWARE DEMONSTRATION COMPLETE!")
        print("=" * 70)
        print("✅ All middleware features demonstrated successfully")
        print("✅ Security, context management, and observability working together")
        print("✅ Production-ready patterns and best practices shown")
        print("✅ Real-world scenarios tested and validated")
        print("\n💡 Key Takeaways:")
        print("• Middleware provides essential production capabilities")
        print("• Multiple middleware can be chained for comprehensive protection")
        print("• Context management prevents token limit issues")
        print("• Security middleware protects against common threats")
        print("• Observability middleware enables monitoring and debugging")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        logger.exception("Demo error details:")


if __name__ == "__main__":
    asyncio.run(main())
