"""
Tests for the evaluation system.

Tests cover:
- Reference-based judges (ExactMatch, FuzzyMatch, Contains)
- Answer extraction strategies
- Composite judges
- Edge cases and error handling
"""

from typing import List

import pytest
from pydantic import BaseModel

from picoagents.agents import Agent
from picoagents.eval import (
    AgentEvalTarget,
    CompositeJudge,
    ContainsJudge,
    EvalRunner,
    ExactMatchJudge,
    FuzzyMatchJudge,
    ModelEvalTarget,
)
from picoagents.llm import BaseChatCompletionClient
from picoagents.messages import AssistantMessage, ToolMessage, UserMessage
from picoagents.types import ChatCompletionResult, Task, RunTrajectory, Usage


class MockChatCompletionClient(BaseChatCompletionClient):
    """Mock client for testing."""

    def __init__(self, model: str = "test-model", response: str = "Test response"):
        super().__init__(model=model)
        self.response = response
        self.call_count = 0

    async def create(self, messages: List, tools=None, output_format=None, **kwargs):
        self.call_count += 1
        return ChatCompletionResult(
            message=AssistantMessage(content=self.response, source="mock"),
            usage=Usage(
                duration_ms=100, llm_calls=1, tokens_input=50, tokens_output=25
            ),
            model=self.model,
            finish_reason="stop",
        )

    async def create_stream(
        self, messages: List, tools=None, output_format=None, **kwargs
    ):
        """Mock streaming implementation."""
        # For simplicity, just yield the final result
        from picoagents.types import ChatCompletionChunk

        yield ChatCompletionChunk(
            content=self.response, is_complete=True, tool_call_chunk=None,
            usage=Usage(duration_ms=100, llm_calls=1, tokens_input=50, tokens_output=25)
        )


def create_test_trajectory(
    task: Task, messages: List, success: bool = True, error: str | None = None
) -> RunTrajectory:
    """Helper to create test trajectories."""
    return RunTrajectory(
        task=task,
        messages=messages,
        success=success,
        error=error,
        usage=Usage(duration_ms=100, llm_calls=1, tokens_input=50, tokens_output=25),
        metadata={},
    )


@pytest.mark.asyncio
async def test_exact_match_judge_success():
    """Test ExactMatchJudge with matching answer."""
    judge = ExactMatchJudge()

    task = Task(name="Math", input="What is 2 + 2?", expected_output="4")

    trajectory = create_test_trajectory(
        task,
        messages=[
            UserMessage(content="What is 2 + 2?", source="user"),
            AssistantMessage(content="4", source="agent"),
        ],
    )

    score = await judge.score(trajectory)

    assert score.overall == 10.0
    assert score.dimensions["accuracy"] == 10.0
    assert score.metadata["match"] is True


@pytest.mark.asyncio
async def test_exact_match_judge_failure():
    """Test ExactMatchJudge with non-matching answer."""
    judge = ExactMatchJudge()

    task = Task(name="Math", input="What is 2 + 2?", expected_output="4")

    trajectory = create_test_trajectory(
        task,
        messages=[
            UserMessage(content="What is 2 + 2?", source="user"),
            AssistantMessage(content="5", source="agent"),
        ],
    )

    score = await judge.score(trajectory)

    assert score.overall == 0.0
    assert score.dimensions["accuracy"] == 0.0
    assert score.metadata["match"] is False


@pytest.mark.asyncio
async def test_exact_match_case_insensitive():
    """Test ExactMatchJudge case insensitivity."""
    judge = ExactMatchJudge(case_sensitive=False)

    task = Task(name="Capital", input="Capital of France?", expected_output="paris")

    trajectory = create_test_trajectory(
        task,
        messages=[
            UserMessage(content="Capital of France?", source="user"),
            AssistantMessage(content="Paris", source="agent"),
        ],
    )

    score = await judge.score(trajectory)

    assert score.overall == 10.0
    assert score.metadata["match"] is True


@pytest.mark.asyncio
async def test_answer_extraction_last_non_empty():
    """Test last_non_empty extraction strategy."""
    judge = ExactMatchJudge(answer_strategy="last_non_empty")

    task = Task(name="Test", input="Question", expected_output="42")

    trajectory = create_test_trajectory(
        task,
        messages=[
            UserMessage(content="Question", source="user"),
            AssistantMessage(content="42", source="agent"),
            AssistantMessage(content="", source="agent"),  # Empty message at end
        ],
    )

    score = await judge.score(trajectory)

    assert score.overall == 10.0  # Should find "42" before empty message


@pytest.mark.asyncio
async def test_answer_extraction_last_assistant():
    """Test last_assistant extraction strategy (skips tool results)."""
    judge = ExactMatchJudge(answer_strategy="last_assistant")

    task = Task(name="Tool Task", input="Use tool", expected_output="Result")

    trajectory = create_test_trajectory(
        task,
        messages=[
            UserMessage(content="Use tool", source="user"),
            AssistantMessage(content="Result", source="agent"),
            ToolMessage(
                content="Tool output",
                source="tool",
                tool_call_id="123",
                tool_name="test_tool",
                success=True,
            ),
        ],
    )

    score = await judge.score(trajectory)

    assert score.overall == 10.0  # Should find "Result", not tool output


@pytest.mark.asyncio
async def test_fuzzy_match_judge():
    """Test FuzzyMatchJudge with similar strings."""
    judge = FuzzyMatchJudge(threshold=0.8)

    task = Task(
        name="Description",
        input="Describe Paris",
        expected_output="Paris is the capital of France",
    )

    trajectory = create_test_trajectory(
        task,
        messages=[
            UserMessage(content="Describe Paris", source="user"),
            AssistantMessage(content="Paris is France's capital", source="agent"),
        ],
    )

    score = await judge.score(trajectory)

    # Should get high score due to similarity
    assert score.overall > 5.0
    assert "similarity" in score.metadata
    assert 0 <= score.metadata["similarity"] <= 1.0


@pytest.mark.asyncio
async def test_contains_judge_success():
    """Test ContainsJudge when expected is in response."""
    judge = ContainsJudge()

    task = Task(
        name="Capital", input="What is the capital of France?", expected_output="Paris"
    )

    trajectory = create_test_trajectory(
        task,
        messages=[
            UserMessage(content="What is the capital of France?", source="user"),
            AssistantMessage(
                content="The capital of France is Paris, a beautiful city.",
                source="agent",
            ),
        ],
    )

    score = await judge.score(trajectory)

    assert score.overall == 10.0
    assert score.metadata["contains"] is True


@pytest.mark.asyncio
async def test_contains_judge_failure():
    """Test ContainsJudge when expected is not in response."""
    judge = ContainsJudge()

    task = Task(
        name="Capital", input="What is the capital of France?", expected_output="Paris"
    )

    trajectory = create_test_trajectory(
        task,
        messages=[
            UserMessage(content="What is the capital of France?", source="user"),
            AssistantMessage(content="France is a beautiful country.", source="agent"),
        ],
    )

    score = await judge.score(trajectory)

    assert score.overall == 0.0
    assert score.metadata["contains"] is False


@pytest.mark.asyncio
async def test_composite_judge():
    """Test CompositeJudge combining multiple judges."""
    exact_judge = ExactMatchJudge()
    contains_judge = ContainsJudge()

    composite = CompositeJudge([(exact_judge, 0.7), (contains_judge, 0.3)])

    task = Task(name="Test", input="Question", expected_output="Answer")

    trajectory = create_test_trajectory(
        task,
        messages=[
            UserMessage(content="Question", source="user"),
            AssistantMessage(content="The Answer is correct", source="agent"),
        ],
    )

    score = await composite.score(trajectory)

    # Exact match fails (0), contains succeeds (10)
    # Score should be: 0.7 * 0 + 0.3 * 10 = 3.0
    assert score.overall == pytest.approx(3.0)
    assert "sub_judges" in score.metadata
    assert len(score.metadata["sub_judges"]) == 2


@pytest.mark.asyncio
async def test_composite_judge_normalization():
    """Test CompositeJudge weight normalization."""
    judge1 = ExactMatchJudge()
    judge2 = ContainsJudge()

    # Weights don't sum to 1.0, but normalization is on
    composite = CompositeJudge([(judge1, 2.0), (judge2, 3.0)], normalize_weights=True)

    task = Task(name="Test", input="Question", expected_output="Answer")

    trajectory = create_test_trajectory(
        task,
        messages=[
            UserMessage(content="Question", source="user"),
            AssistantMessage(content="Answer", source="agent"),
        ],
    )

    score = await composite.score(trajectory)

    # Both match (10), normalized weights: 2/5 and 3/5
    # Score: 0.4 * 10 + 0.6 * 10 = 10.0
    assert score.overall == 10.0


@pytest.mark.asyncio
async def test_judge_with_failed_trajectory():
    """Test judges handle failed trajectories gracefully."""
    judge = ExactMatchJudge()

    task = Task(name="Test", input="Question", expected_output="Answer")

    trajectory = create_test_trajectory(
        task, messages=[], success=False, error="Execution failed"
    )

    score = await judge.score(trajectory)

    assert score.overall == 0.0
    assert "error" in score.metadata


@pytest.mark.asyncio
async def test_judge_missing_expected_output():
    """Test judges raise error when expected_output is missing."""
    judge = ExactMatchJudge()

    task = Task(
        name="Test",
        input="Question",
        expected_output=None,
        
    )

    trajectory = create_test_trajectory(
        task,
        messages=[
            UserMessage(content="Question", source="user"),
            AssistantMessage(content="Answer", source="agent"),
        ],
    )

    with pytest.raises(ValueError, match="expected_output"):
        await judge.score(trajectory)


@pytest.mark.asyncio
async def test_eval_runner_with_agent():
    """Test EvalRunner with agent target."""
    client = MockChatCompletionClient(response="4")
    agent = Agent(
        name="test-agent",
        description="Test agent",
        instructions="Answer questions",
        model_client=client,
    )

    target = AgentEvalTarget(agent)
    judge = ExactMatchJudge()
    runner = EvalRunner(judge, parallel_tasks=False)

    tasks = [
        Task(name="Math1", input="2+2", expected_output="4"),
        Task(
            name="Math2", input="3+3", expected_output="6"
        ),  # Correct answer for 3+3
    ]

    scores = await runner.evaluate(target, tasks)

    assert len(scores) == 2
    assert scores[0].overall == 10.0  # Correct (4 == 4)
    assert scores[1].overall == 0.0  # Wrong (4 != 6)


@pytest.mark.asyncio
async def test_answer_extraction_all_assistant():
    """Test all_assistant extraction (concatenates all AssistantMessages)."""
    judge = ExactMatchJudge(answer_strategy="all_assistant")

    task = Task(
        name="Multi-turn",
        input="Tell me about Paris",
        expected_output="Paris is the capital of France.\nIt is beautiful.",  # Newline separator
    )

    trajectory = create_test_trajectory(
        task,
        messages=[
            UserMessage(content="Tell me about Paris", source="user"),
            AssistantMessage(content="Paris is the capital of France.", source="agent"),
            UserMessage(content="Tell me more", source="user"),
            AssistantMessage(content="It is beautiful.", source="agent"),
        ],
    )

    score = await judge.score(trajectory)

    # Should concatenate both assistant messages with newline
    assert score.overall == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
