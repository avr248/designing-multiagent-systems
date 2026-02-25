"""
Main CLI entry point for PicoAgents.

Provides a unified command-line interface with subcommands for different functionality.
"""

import argparse
import asyncio
import sys
from typing import List, Optional


def main(args: Optional[List[str]] = None) -> None:
    """Main CLI entry point with subcommands.

    Args:
        args: Optional list of arguments (for testing)
    """
    parser = argparse.ArgumentParser(
        prog="picoagents",
        description="PicoAgents - Lightweight AI agent framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  ui          Launch web interface for agents/orchestrators/workflows
  eval        Run evaluations to compare agent configurations

Examples:
  picoagents ui                              # Launch UI for current directory
  picoagents ui --dir ./agents               # Launch UI for specific directory
  picoagents eval list                       # List available datasets
  picoagents eval run coding_v1              # Run evaluation with dataset
        """,
    )

    # Add version flag
    parser.add_argument("--version", action="version", version="picoagents 0.1.0")

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="<command>",
    )

    # UI subcommand
    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch web interface",
        description="Launch PicoAgents web interface for interacting with entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  picoagents ui                    # Scan current directory
  picoagents ui --dir ./agents     # Scan specific directory
  picoagents ui --port 8000        # Use different port
  picoagents ui --no-open          # Don't open browser
  picoagents ui --reload           # Enable auto-reload for development
        """,
    )

    ui_parser.add_argument(
        "--dir",
        default=".",
        help="Directory to scan for agents, orchestrators, and workflows (default: current directory)",
    )
    ui_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Port to run server on (default: 8080)",
    )
    ui_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind server to (default: 127.0.0.1)",
    )
    ui_parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't automatically open browser",
    )
    ui_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    ui_parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)",
    )

    # Eval subcommand
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run evaluations to compare agent configurations",
        description="Run evaluation datasets against multiple agent configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  picoagents eval list                          # List built-in datasets
  picoagents eval run coding_v1                 # Run built-in dataset
  picoagents eval run ./my_dataset.json         # Run custom dataset
  picoagents eval run dataset.json -c configs.json  # With config file
  picoagents eval results                       # List saved results
  picoagents eval results ./results/run_123.json    # View specific result
        """,
    )

    eval_subparsers = eval_parser.add_subparsers(
        dest="eval_command",
        help="Eval commands",
        metavar="<action>",
    )

    # eval list - list available datasets
    eval_subparsers.add_parser(
        "list",
        help="List available evaluation datasets",
    )

    # eval run - run an evaluation
    run_parser = eval_subparsers.add_parser(
        "run",
        help="Run an evaluation dataset",
    )
    run_parser.add_argument(
        "dataset",
        help="Dataset name (built-in) or path to JSON file",
    )
    run_parser.add_argument(
        "-c", "--configs",
        help="Path to JSON file with agent configurations",
    )
    run_parser.add_argument(
        "-o", "--output",
        help="Output directory for results (default: .picoagents/eval)",
        default=".picoagents/eval",
    )
    run_parser.add_argument(
        "--baseline",
        help="Target name to use as baseline for comparison",
    )
    run_parser.add_argument(
        "--parallel-tasks",
        action="store_true",
        help="Run tasks in parallel (may affect fairness)",
    )
    run_parser.add_argument(
        "--parallel-targets",
        action="store_true",
        help="Run targets in parallel",
    )
    run_parser.add_argument(
        "--task-filter",
        help="Filter tasks by category (e.g., 'coding')",
    )

    # eval results - view results
    results_parser = eval_subparsers.add_parser(
        "results",
        help="List or view evaluation results",
    )
    results_parser.add_argument(
        "path",
        nargs="?",
        help="Path to specific results file (omit to list all)",
    )
    results_parser.add_argument(
        "--dir",
        default=".picoagents/eval",
        help="Directory containing results (default: .picoagents/eval)",
    )
    results_parser.add_argument(
        "--show-breakdown",
        action="store_true",
        help="Show per-task breakdown",
    )
    results_parser.add_argument(
        "--show-files",
        action="store_true",
        help="Show file read analysis",
    )

    # Parse arguments
    parsed_args = parser.parse_args(args)

    # Handle no command provided
    if parsed_args.command is None:
        parser.print_help()
        print("\nTip: Try 'picoagents ui' to launch the web interface")
        sys.exit(1)

    # Route to appropriate handler
    if parsed_args.command == "ui":
        _handle_ui_command(parsed_args)
    elif parsed_args.command == "eval":
        _handle_eval_command(parsed_args, eval_parser)
    else:
        parser.print_help()
        sys.exit(1)


def _handle_ui_command(args: argparse.Namespace) -> None:
    """Handle the 'ui' subcommand."""
    try:
        from ..webui import webui

        webui(
            entities_dir=args.dir,
            port=args.port,
            host=args.host,
            auto_open=not args.no_open,
            reload=args.reload,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        print("\nShutting down PicoAgents UI")
        sys.exit(0)
    except ImportError as e:
        print(f"Error importing WebUI: {e}")
        print("Make sure to install web dependencies: pip install picoagents[web]")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting UI: {e}")
        sys.exit(1)


def _handle_eval_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Handle the 'eval' subcommand."""
    if args.eval_command is None:
        parser.print_help()
        sys.exit(1)

    if args.eval_command == "list":
        _eval_list()
    elif args.eval_command == "run":
        _eval_run(args)
    elif args.eval_command == "results":
        _eval_results(args)
    else:
        parser.print_help()
        sys.exit(1)


def _eval_list() -> None:
    """List available evaluation datasets."""
    try:
        from ..eval import list_builtin_datasets

        datasets = list_builtin_datasets()

        print("Available Evaluation Datasets")
        print("=" * 50)

        if not datasets:
            print("No built-in datasets found.")
            return

        for name in datasets:
            print(f"  - {name}")

        print()
        print("Usage:")
        print("  picoagents eval run <dataset_name>")
        print("  picoagents eval run ./path/to/custom.json")

    except ImportError as e:
        print(f"Error importing eval module: {e}")
        sys.exit(1)


def _eval_run(args: argparse.Namespace) -> None:
    """Run an evaluation."""
    import json
    import os
    from pathlib import Path

    try:
        from ..eval import (
            AgentConfig,
            Dataset,
            EvalRunner,
            PicoAgentTarget,
            load_builtin_dataset,
            print_results,
        )

        # Load dataset
        dataset_path = args.dataset
        if os.path.exists(dataset_path):
            print(f"Loading dataset from: {dataset_path}")
            dataset = Dataset.from_json(dataset_path)
        else:
            print(f"Loading built-in dataset: {dataset_path}")
            try:
                dataset = load_builtin_dataset(dataset_path)
            except FileNotFoundError:
                print(f"Dataset not found: {dataset_path}")
                print("Use 'picoagents eval list' to see available datasets")
                sys.exit(1)

        print(f"Dataset: {dataset.name} ({len(list(dataset.tasks))} tasks)")

        # Load or create configurations
        configs: List[AgentConfig] = []
        if args.configs:
            print(f"Loading configurations from: {args.configs}")
            with open(args.configs) as f:
                config_data = json.load(f)
            configs = [AgentConfig.from_dict(c) for c in config_data]
        else:
            print("Using default configurations (baseline vs head_tail)")
            configs = [
                AgentConfig(name="baseline", compaction=None),
                AgentConfig(name="head_tail", compaction="head_tail"),
            ]

        print(f"Configurations: {[c.name for c in configs]}")

        # Create targets
        targets = [PicoAgentTarget(config) for config in configs]

        # Create judge
        print("\nNote: Full evaluation requires a configured LLM judge.")
        print("   For now, results will show metrics without scoring.")
        print()

        # Create runner
        runner = EvalRunner(
            judge=_create_mock_judge(),
            parallel_tasks=args.parallel_tasks,
            parallel_targets=args.parallel_targets,
        )

        # Apply task filter if specified
        task_filter = None
        if args.task_filter:
            category = args.task_filter
            task_filter = lambda t: t.category == category
            print(f"Filtering tasks by category: {category}")

        # Run evaluation
        print("\nRunning evaluation...")
        results = asyncio.run(
            runner.run(dataset, targets, task_filter=task_filter)
        )

        # Print results
        print("\n")
        print_results(
            results,
            baseline=args.baseline or configs[0].name,
            show_task_breakdown=True,
            show_file_analysis=True,
        )

        # Save results
        output_path = results.save()
        print(f"\nResults saved to: {output_path}")

    except ImportError as e:
        print(f"Error importing eval module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _eval_results(args: argparse.Namespace) -> None:
    """List or view evaluation results."""
    from pathlib import Path

    try:
        from ..eval import (
            list_eval_results,
            load_eval_results,
            print_results,
        )

        if args.path:
            print(f"Loading results from: {args.path}")
            results = load_eval_results(args.path)
            print_results(
                results,
                show_task_breakdown=args.show_breakdown,
                show_file_analysis=args.show_files,
            )
        else:
            results_dir = Path(args.dir)
            if not results_dir.exists():
                print(f"No results directory found: {results_dir}")
                return

            result_files = list_eval_results(results_dir)

            if not result_files:
                print(f"No evaluation results found in: {results_dir}")
                return

            print("Evaluation Results")
            print("=" * 50)
            for result_file in result_files:
                print(f"  - {result_file}")

            print()
            print("View a result:")
            print("  picoagents eval results <path_to_result.json>")

    except ImportError as e:
        print(f"Error importing eval module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def _create_mock_judge():
    """Create a mock judge for CLI usage when no LLM is configured."""
    from ..eval._base import EvalJudge
    from ..types import EvalScore, RunTrajectory

    class MockJudge(EvalJudge):
        """Mock judge that returns placeholder scores."""

        def __init__(self):
            super().__init__(name="mock_judge")

        async def score(
            self,
            trajectory: RunTrajectory,
            criteria: Optional[List[str]] = None,
            cancellation_token=None,
        ) -> EvalScore:
            criteria = criteria or ["task_completion"]
            return EvalScore(
                overall=0.0,
                dimensions={c: 0.0 for c in criteria},
                reasoning={c: "Mock judge - no LLM configured" for c in criteria},
                trajectory=trajectory,
                metadata={"mock": True},
            )

    return MockJudge()


if __name__ == "__main__":
    main()
