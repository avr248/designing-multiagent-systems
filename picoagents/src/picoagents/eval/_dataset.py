"""
Dataset definitions for evaluation.

This module defines Dataset - a collection of tasks with evaluation criteria
that can be used to evaluate and compare targets.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..types import Task


@dataclass
class Dataset:
    """A collection of tasks for evaluation.

    Datasets group related tasks together and provide common evaluation
    criteria. They can be loaded from JSON files or defined in code.

    Example:
        >>> dataset = Dataset.from_json("coding_v1.json")
        >>> for task in dataset.tasks:
        ...     print(task.id)
    """

    # Dataset identification
    name: str
    version: str = "1.0.0"
    description: str = ""

    # Tasks
    tasks: List[Task] = field(default_factory=list)

    # Dataset-level settings
    categories: List[str] = field(default_factory=list)
    default_eval_criteria: List[str] = field(default_factory=lambda: ["task_completion"])

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Populate categories from tasks if not provided."""
        if not self.categories:
            self.categories = list(set(t.category for t in self.tasks))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize dataset to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "categories": self.categories,
            "default_eval_criteria": self.default_eval_criteria,
            "tasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "input": t.input,
                    "category": t.category,
                    "eval_criteria": t.eval_criteria,
                    "expected_output": t.expected_output,
                    "rubric": t.rubric,
                    "metadata": t.metadata,
                }
                for t in self.tasks
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dataset":
        """Create dataset from dictionary."""
        tasks = [
            Task(
                name=t["name"],
                input=t["input"],
                id=t.get("id"),
                category=t.get("category", "general"),
                eval_criteria=t.get("eval_criteria", ["task_completion"]),
                expected_output=t.get("expected_output"),
                rubric=t.get("rubric", {}),
                metadata=t.get("metadata", {}),
            )
            for t in data.get("tasks", [])
        ]
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            tasks=tasks,
            categories=data.get("categories", []),
            default_eval_criteria=data.get("default_eval_criteria", ["task_completion"]),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: Path) -> "Dataset":
        """Load dataset from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Dataset instance
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: Path) -> None:
        """Save dataset to JSON file.

        Args:
            path: Path to write JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def filter_by_category(self, category: str) -> "Dataset":
        """Return subset of tasks matching category."""
        filtered_tasks = [t for t in self.tasks if t.category == category]
        return Dataset(
            name=f"{self.name}_{category}",
            version=self.version,
            description=f"{self.description} (filtered: {category})",
            tasks=filtered_tasks,
            categories=[category],
            default_eval_criteria=self.default_eval_criteria,
            metadata={**self.metadata, "filtered_from": self.name},
        )

    def filter_by_ids(self, task_ids: List[str]) -> "Dataset":
        """Return subset of tasks matching IDs."""
        filtered_tasks = [t for t in self.tasks if t.id in task_ids]
        return Dataset(
            name=f"{self.name}_subset",
            version=self.version,
            description=f"{self.description} (subset)",
            tasks=filtered_tasks,
            categories=list(set(t.category for t in filtered_tasks)),
            default_eval_criteria=self.default_eval_criteria,
            metadata={**self.metadata, "filtered_from": self.name},
        )

    def filter(self, predicate: Callable[[Task], bool]) -> "Dataset":
        """Return subset of tasks matching predicate."""
        filtered_tasks = [t for t in self.tasks if predicate(t)]
        return Dataset(
            name=f"{self.name}_filtered",
            version=self.version,
            description=f"{self.description} (custom filter)",
            tasks=filtered_tasks,
            categories=list(set(t.category for t in filtered_tasks)),
            default_eval_criteria=self.default_eval_criteria,
            metadata={**self.metadata, "filtered_from": self.name},
        )

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __repr__(self) -> str:
        return f"Dataset(name={self.name!r}, tasks={len(self.tasks)}, categories={self.categories})"


def load_builtin_dataset(name: str) -> Dataset:
    """Load a built-in evaluation dataset.

    Args:
        name: Dataset name (e.g., "coding_v1")

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset not found
    """
    datasets_dir = Path(__file__).parent / "datasets"

    # Try exact name first
    path = datasets_dir / f"{name}.json"
    if path.exists():
        return Dataset.from_json(path)

    # List available datasets
    available = [p.stem for p in datasets_dir.glob("*.json")]
    raise ValueError(f"Dataset '{name}' not found. Available: {available}")


def list_builtin_datasets() -> List[str]:
    """List available built-in datasets.

    Returns:
        List of dataset names
    """
    datasets_dir = Path(__file__).parent / "datasets"
    if not datasets_dir.exists():
        return []
    return [p.stem for p in datasets_dir.glob("*.json")]
