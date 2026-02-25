"""
Evaluation analysis utilities.

This module provides functions for analyzing and displaying evaluation results.
"""

from typing import Any, Dict, Optional

from ._results import EvalResults, TargetSummary


def format_summary_table(results: EvalResults, baseline: Optional[str] = None) -> str:
    """Format results as a summary table."""
    summaries = results.get_summaries()
    comparison = results.compare_targets(baseline)

    if not summaries:
        return "No results to display."

    lines = [
        f"Evaluation: {results.dataset_name} ({len(results.task_ids)} tasks)",
        f"Run ID: {results.run_id}",
        f"Date: {results.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
    ]

    header = f"{'Target':<20} {'Score':>8} {'Tokens':>12} {'Iters':>8} {'Duration':>10} {'Files':>8} {'Dupes':>8}"
    lines.append(header)
    lines.append("-" * 80)

    for target_name in results.target_names:
        summary = summaries.get(target_name)
        if not summary:
            continue

        comp = comparison.get(target_name, {})

        score_str = f"{summary.avg_score:.1f}"
        tokens_str = f"{summary.total_tokens:,}"
        iters_str = f"{summary.total_iterations}"
        duration_str = f"{summary.total_duration_ms / 1000:.1f}s"
        files_str = f"{summary.total_unique_files}"
        dupes_str = f"{summary.total_duplicate_reads}"

        if not comp.get("is_baseline", False):
            token_pct = comp.get("token_diff_pct", 0)
            if token_pct != 0:
                tokens_str += f" ({token_pct:+.0f}%)"

        row = f"{target_name:<20} {score_str:>8} {tokens_str:>12} {iters_str:>8} {duration_str:>10} {files_str:>8} {dupes_str:>8}"
        lines.append(row)

    lines.append("")

    if baseline and len(results.target_names) > 1:
        lines.append("vs " + (baseline or results.target_names[0]) + ":")
        for target_name in results.target_names:
            comp = comparison.get(target_name, {})
            if comp.get("is_baseline", False):
                continue

            token_diff = comp.get("token_diff_pct", 0)
            score_diff = comp.get("score_diff", 0)
            iter_diff = comp.get("iteration_diff_pct", 0)

            lines.append(
                f"  {target_name}: "
                f"{token_diff:+.1f}% tokens, "
                f"{iter_diff:+.1f}% iters, "
                f"{score_diff:+.2f} score"
            )

    return "\n".join(lines)


def format_task_breakdown(results: EvalResults) -> str:
    """Format per-task breakdown."""
    lines = [
        "Per-Task Breakdown",
        "=" * 80,
        "",
    ]

    header = f"{'Task':<25}"
    for target_name in results.target_names:
        header += f" {target_name[:15]:>15}"
    lines.append(header)
    lines.append("-" * 80)

    for task_id in results.task_ids:
        lines.append(f"\n{task_id}")

        tokens_row = f"  {'tokens':<23}"
        for target_name in results.target_names:
            result = results.get_result(target_name, task_id)
            if result:
                tokens_row += f" {result.total_tokens:>15,}"
            else:
                tokens_row += f" {'-':>15}"
        lines.append(tokens_row)

        score_row = f"  {'score':<23}"
        for target_name in results.target_names:
            result = results.get_result(target_name, task_id)
            if result:
                score_row += f" {result.score.overall:>15.1f}"
            else:
                score_row += f" {'-':>15}"
        lines.append(score_row)

        iter_row = f"  {'iterations':<23}"
        for target_name in results.target_names:
            result = results.get_result(target_name, task_id)
            if result:
                iter_row += f" {result.iterations:>15}"
            else:
                iter_row += f" {'-':>15}"
        lines.append(iter_row)

    return "\n".join(lines)


def format_file_read_analysis(results: EvalResults) -> str:
    """Format file read pattern analysis."""
    lines = [
        "File Read Analysis",
        "=" * 80,
        "",
    ]

    for target_name in results.target_names:
        lines.append(f"\n{target_name.upper()}")
        lines.append("-" * 40)

        all_reads: Dict[str, int] = {}
        for task_id in results.task_ids:
            result = results.get_result(target_name, task_id)
            if result:
                for path, count in result.files_read.items():
                    all_reads[path] = all_reads.get(path, 0) + count

        if not all_reads:
            lines.append("  No file reads recorded")
            continue

        sorted_reads = sorted(all_reads.items(), key=lambda x: -x[1])

        for path, count in sorted_reads[:20]:
            display_path = path
            if len(path) > 45:
                display_path = "..." + path[-42:]

            bar = "#" * min(count, 20)
            marker = " !!" if count > 1 else ""
            lines.append(f"  {count:>3}x {display_path:<45} {bar}{marker}")

        if len(sorted_reads) > 20:
            lines.append(f"  ... and {len(sorted_reads) - 20} more files")

        total = sum(all_reads.values())
        unique = len(all_reads)
        duplicates = total - unique
        lines.append(f"\n  Total: {total} reads, {unique} unique files")
        if duplicates > 0:
            overhead = duplicates / total * 100
            lines.append(f"  Duplicate reads: {duplicates} ({overhead:.0f}% overhead)")

    return "\n".join(lines)


def format_token_growth(results: EvalResults, task_id: Optional[str] = None) -> str:
    """Format token growth across iterations."""
    if not task_id and results.task_ids:
        task_id = results.task_ids[0]

    if not task_id:
        return "No tasks to analyze."

    lines = [
        f"Token Growth: {task_id}",
        "=" * 80,
        "",
    ]

    for target_name in results.target_names:
        result = results.get_result(target_name, task_id)
        if not result:
            continue

        lines.append(f"\n{target_name}:")

        token_growth = result.metrics.get("token_growth", [])
        if not token_growth:
            lines.append("  No iteration data available")
            continue

        max_tokens = max(t[1] for t in token_growth) if token_growth else 1

        for idx, tokens in token_growth[:15]:
            bar_len = int(tokens / max_tokens * 30)
            bar = "#" * bar_len
            lines.append(f"  {idx:>2}: {tokens:>6,} {bar}")

        if len(token_growth) > 15:
            lines.append(f"  ... ({len(token_growth) - 15} more iterations)")

    return "\n".join(lines)


def print_results(
    results: EvalResults,
    baseline: Optional[str] = None,
    show_task_breakdown: bool = True,
    show_file_analysis: bool = False,
    show_token_growth: bool = False,
) -> None:
    """Print formatted evaluation results."""
    print(format_summary_table(results, baseline))

    if show_task_breakdown:
        print("\n")
        print(format_task_breakdown(results))

    if show_file_analysis:
        print("\n")
        print(format_file_read_analysis(results))

    if show_token_growth:
        print("\n")
        print(format_token_growth(results))
