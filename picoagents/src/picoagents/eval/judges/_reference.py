"""
Reference-based evaluation judges that compare outputs to expected results.

These judges provide deterministic, efficient evaluation for tasks with known
correct answers, without requiring expensive LLM calls.
"""

from difflib import SequenceMatcher
from typing import List, Optional

from ..._cancellation_token import CancellationToken
from ...types import EvalScore, RunTrajectory
from ._base import BaseEvalJudge


class ExactMatchJudge(BaseEvalJudge):
    """Reference-based judge using exact string matching.

    This judge compares the agent's final response to the expected_output
    field in the evaluation task. It's ideal for tasks with deterministic
    answers like math problems, factual questions, or code outputs.

    Example:
        Task: "What is 2 + 2?"
        Expected: "4"
        Agent Response: "4"
        Score: 10.0/10
    """

    def __init__(
        self,
        name: str = "ExactMatch",
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
        answer_strategy: str = "last_non_empty",
    ):
        """Initialize the exact match judge.

        Args:
            name: Human-readable name for this judge
            case_sensitive: Whether to perform case-sensitive matching
            strip_whitespace: Whether to strip leading/trailing whitespace
            answer_strategy: How to extract answer from trajectory
                - "last_non_empty": Last message with content (default)
                - "last_assistant": Last AssistantMessage only
                - "all_assistant": Concatenate all assistant messages
        """
        super().__init__(name, answer_strategy)
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    async def score(
        self,
        trajectory: RunTrajectory,
        criteria: Optional[List[str]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> EvalScore:
        """Score trajectory by comparing to expected output.

        Args:
            trajectory: The execution trajectory to score
            criteria: Ignored for reference-based judges
            cancellation_token: Optional token to cancel scoring

        Returns:
            EvalScore with 10.0 for exact match, 0.0 otherwise

        Raises:
            ValueError: If task doesn't have expected_output
        """
        if not trajectory.task.expected_output:
            raise ValueError(
                f"ExactMatchJudge requires tasks with expected_output. "
                f"Task '{trajectory.task.name}' has no expected_output."
            )

        if not trajectory.success or not trajectory.messages:
            return EvalScore(
                overall=0.0,
                dimensions={"accuracy": 0.0},
                reasoning={
                    "accuracy": f"Execution failed: {trajectory.error or 'No messages generated'}"
                },
                trajectory=trajectory,
                metadata={
                    "judge": self.name,
                    "match": False,
                    "error": trajectory.error,
                },
            )

        actual = self.extract_answer(trajectory)
        expected = trajectory.task.expected_output

        if self.strip_whitespace:
            actual = actual.strip()
            expected = expected.strip()

        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        is_match = actual == expected
        score = 10.0 if is_match else 0.0

        reasoning = (
            f"Expected: '{trajectory.task.expected_output}'\n"
            f"Got: '{actual}'\n"
            f"Match: {is_match}\n"
            f"Extraction strategy: {self.answer_strategy}"
        )

        return EvalScore(
            overall=score,
            dimensions={"accuracy": score},
            reasoning={"accuracy": reasoning},
            trajectory=trajectory,
            metadata={
                "judge": self.name,
                "match": is_match,
                "case_sensitive": self.case_sensitive,
                "strip_whitespace": self.strip_whitespace,
                "answer_strategy": self.answer_strategy,
            },
        )


class FuzzyMatchJudge(BaseEvalJudge):
    """Reference-based judge using fuzzy string matching.

    This judge uses sequence matching to score similarity between the
    agent's response and expected output. Useful for tasks where responses
    might vary slightly in formatting or wording.

    Example:
        Expected: "The capital of France is Paris"
        Agent: "Paris is the capital of France"
        Score: 9.5/10 (high similarity despite word order)
    """

    def __init__(
        self,
        name: str = "FuzzyMatch",
        threshold: float = 0.8,
        case_sensitive: bool = False,
        answer_strategy: str = "last_non_empty",
    ):
        """Initialize the fuzzy match judge.

        Args:
            name: Human-readable name for this judge
            threshold: Minimum similarity ratio (0-1) for full score
            case_sensitive: Whether to perform case-sensitive matching
            answer_strategy: How to extract answer from trajectory
        """
        super().__init__(name, answer_strategy)
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        self.threshold = threshold
        self.case_sensitive = case_sensitive

    async def score(
        self,
        trajectory: RunTrajectory,
        criteria: Optional[List[str]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> EvalScore:
        """Score trajectory using fuzzy string matching.

        Args:
            trajectory: The execution trajectory to score
            criteria: Ignored for reference-based judges
            cancellation_token: Optional token to cancel scoring

        Returns:
            EvalScore with similarity-based score (0-10 scale)

        Raises:
            ValueError: If task doesn't have expected_output
        """
        if not trajectory.task.expected_output:
            raise ValueError(
                f"FuzzyMatchJudge requires tasks with expected_output. "
                f"Task '{trajectory.task.name}' has no expected_output."
            )

        if not trajectory.success or not trajectory.messages:
            return EvalScore(
                overall=0.0,
                dimensions={"accuracy": 0.0},
                reasoning={
                    "accuracy": f"Execution failed: {trajectory.error or 'No messages generated'}"
                },
                trajectory=trajectory,
                metadata={
                    "judge": self.name,
                    "similarity": 0.0,
                    "error": trajectory.error,
                },
            )

        actual = self.extract_answer(trajectory)
        expected = trajectory.task.expected_output.strip()

        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        similarity = SequenceMatcher(None, actual, expected).ratio()

        score = (
            (similarity / self.threshold) * 10.0
            if similarity <= self.threshold
            else 10.0
        )
        score = min(score, 10.0)

        reasoning = (
            f"Expected: '{trajectory.task.expected_output}'\n"
            f"Got: '{actual}'\n"
            f"Similarity: {similarity:.2%} (threshold: {self.threshold:.2%})\n"
            f"Extraction strategy: {self.answer_strategy}"
        )

        return EvalScore(
            overall=score,
            dimensions={"accuracy": score},
            reasoning={"accuracy": reasoning},
            trajectory=trajectory,
            metadata={
                "judge": self.name,
                "similarity": similarity,
                "threshold": self.threshold,
                "case_sensitive": self.case_sensitive,
                "answer_strategy": self.answer_strategy,
            },
        )


class ContainsJudge(BaseEvalJudge):
    """Reference-based judge that checks if expected output is contained in response.

    Useful for tasks where the agent might provide additional context beyond
    the expected answer.

    Example:
        Expected: "Paris"
        Agent: "The capital of France is Paris, a beautiful city."
        Score: 10.0/10 (contains the expected answer)
    """

    def __init__(
        self,
        name: str = "Contains",
        case_sensitive: bool = False,
        answer_strategy: str = "last_non_empty",
    ):
        """Initialize the contains judge.

        Args:
            name: Human-readable name for this judge
            case_sensitive: Whether to perform case-sensitive matching
            answer_strategy: How to extract answer from trajectory
        """
        super().__init__(name, answer_strategy)
        self.case_sensitive = case_sensitive

    async def score(
        self,
        trajectory: RunTrajectory,
        criteria: Optional[List[str]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> EvalScore:
        """Score trajectory by checking if expected output is contained.

        Args:
            trajectory: The execution trajectory to score
            criteria: Ignored for reference-based judges
            cancellation_token: Optional token to cancel scoring

        Returns:
            EvalScore with 10.0 if contains expected, 0.0 otherwise

        Raises:
            ValueError: If task doesn't have expected_output
        """
        if not trajectory.task.expected_output:
            raise ValueError(
                f"ContainsJudge requires tasks with expected_output. "
                f"Task '{trajectory.task.name}' has no expected_output."
            )

        if not trajectory.success or not trajectory.messages:
            return EvalScore(
                overall=0.0,
                dimensions={"accuracy": 0.0},
                reasoning={
                    "accuracy": f"Execution failed: {trajectory.error or 'No messages generated'}"
                },
                trajectory=trajectory,
                metadata={
                    "judge": self.name,
                    "contains": False,
                    "error": trajectory.error,
                },
            )

        actual = self.extract_answer(trajectory)
        expected = trajectory.task.expected_output

        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        contains = expected in actual
        score = 10.0 if contains else 0.0

        reasoning = (
            f"Expected substring: '{trajectory.task.expected_output}'\n"
            f"Agent response: '{actual}'\n"
            f"Contains expected: {contains}\n"
            f"Extraction strategy: {self.answer_strategy}"
        )

        return EvalScore(
            overall=score,
            dimensions={"accuracy": score},
            reasoning={"accuracy": reasoning},
            trajectory=trajectory,
            metadata={
                "judge": self.name,
                "contains": contains,
                "case_sensitive": self.case_sensitive,
                "answer_strategy": self.answer_strategy,
            },
        )
