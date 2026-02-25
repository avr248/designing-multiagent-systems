"""
Base classes for the evaluation system.

This module defines the abstract base classes: Target (what we run tasks against)
and EvalJudge (what scores the results).
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .._cancellation_token import CancellationToken
from ..types import EvalScore, RunTrajectory, Task


class Target(ABC):
    """Abstract base class for anything that can run tasks.

    A target wraps a system under test (agent, model, orchestrator, etc.)
    and provides a uniform interface: give it a Task, get back a RunTrajectory.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def run(
        self, task: Task, cancellation_token: Optional[CancellationToken] = None
    ) -> RunTrajectory:
        """Execute the task and return the complete trajectory.

        Args:
            task: The task to execute
            cancellation_token: Optional token to cancel execution

        Returns:
            RunTrajectory containing the complete execution sequence
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class EvalJudge(ABC):
    """Abstract base class for evaluation judges."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def score(
        self,
        trajectory: RunTrajectory,
        criteria: Optional[List[str]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> EvalScore:
        """Score a run trajectory.

        Args:
            trajectory: The execution trajectory to score
            criteria: Optional list of evaluation dimensions to score.
                      If not provided, uses trajectory.task.eval_criteria,
                      falling back to generic defaults.
            cancellation_token: Optional token to cancel scoring

        Returns:
            EvalScore with overall and dimensional scores
        """
        pass
