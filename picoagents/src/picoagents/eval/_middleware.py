"""
Run middleware for metrics collection.

This module provides RunMiddleware - a middleware that captures
detailed metrics during agent execution for evaluation analysis.
"""

from typing import Any, Dict, List, Optional

from .._middleware import BaseMiddleware, MiddlewareContext


class RunMiddleware(BaseMiddleware):
    """Middleware that captures iteration-level metrics during evaluation runs.

    Tracks:
    - Per-iteration token usage (input/output)
    - Tool calls with parameters and results
    - File read patterns (which files, duplicates)
    - Compaction events
    """

    def __init__(self):
        """Initialize run middleware."""
        self.reset()

    def reset(self) -> None:
        """Reset all collected metrics."""
        self.iterations: List[Dict[str, Any]] = []
        self.current_iteration: Optional[Dict[str, Any]] = None
        self.file_reads: Dict[str, int] = {}  # path -> count
        self.tool_calls: List[Dict[str, Any]] = []
        self.compaction_events: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []

    async def process_request(self, context: MiddlewareContext):
        """Track request phase - start of model calls."""
        if context.operation == "model_call":
            self.current_iteration = {
                "index": len(self.iterations),
                "input_tokens": 0,
                "output_tokens": 0,
                "tool_calls": [],
                "message_count": len(context.data) if isinstance(context.data, list) else 0,
            }

        yield context

    async def process_response(self, context: MiddlewareContext, result: Any):
        """Track response phase - capture metrics."""
        if context.operation == "model_call" and self.current_iteration is not None:
            if hasattr(result, "usage"):
                self.current_iteration["input_tokens"] = getattr(result.usage, "tokens_input", 0)
                self.current_iteration["output_tokens"] = getattr(result.usage, "tokens_output", 0)

            if hasattr(result, "message") and hasattr(result.message, "tool_calls"):
                tool_calls = result.message.tool_calls or []
                self.current_iteration["tool_call_count"] = len(tool_calls)

            self.iterations.append(self.current_iteration)
            self.current_iteration = None

        elif context.operation == "tool_call":
            tool_name = getattr(context.data, "tool_name", "unknown")
            parameters = getattr(context.data, "parameters", {})

            tool_record = {
                "name": tool_name,
                "parameters": parameters,
                "success": getattr(result, "success", True) if result else False,
            }

            if tool_name in ("read_file", "Read", "read"):
                path = parameters.get("path") or parameters.get("file_path") or parameters.get("filename", "unknown")
                self.file_reads[path] = self.file_reads.get(path, 0) + 1
                tool_record["file_path"] = path

            self.tool_calls.append(tool_record)

            if self.current_iteration is not None:
                self.current_iteration["tool_calls"].append(tool_record)

        yield result

    async def process_error(self, context: MiddlewareContext, error: Exception):
        """Track errors."""
        self.errors.append({
            "operation": context.operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
        })

        if False:  # Type hint for async generator
            yield
        raise error

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics.

        Returns:
            Dict with aggregated metrics
        """
        total_input = sum(it.get("input_tokens", 0) for it in self.iterations)
        total_output = sum(it.get("output_tokens", 0) for it in self.iterations)
        unique_files = len(self.file_reads)
        total_reads = sum(self.file_reads.values())
        duplicate_reads = total_reads - unique_files if total_reads > unique_files else 0

        return {
            # Token metrics
            "total_tokens": total_input + total_output,
            "input_tokens": total_input,
            "output_tokens": total_output,

            # Iteration metrics
            "iterations": len(self.iterations),
            "iteration_details": self.iterations,

            # Token growth pattern
            "token_growth": [(it.get("index", i), it.get("input_tokens", 0)) for i, it in enumerate(self.iterations)],

            # Tool metrics
            "tool_calls": len(self.tool_calls),
            "tool_call_details": self.tool_calls,
            "tools_used": list(set(tc["name"] for tc in self.tool_calls)),

            # File access patterns
            "file_reads": self.file_reads,
            "unique_files": unique_files,
            "total_file_reads": total_reads,
            "duplicate_reads": duplicate_reads,
            "duplicate_read_ratio": duplicate_reads / total_reads if total_reads > 0 else 0,

            # Compaction metrics
            "compaction_events": len(self.compaction_events),
            "compaction_details": self.compaction_events,
            "tokens_saved": sum(e.get("tokens_saved", 0) for e in self.compaction_events),

            # Errors
            "errors": self.errors,
            "error_count": len(self.errors),
        }

    def record_compaction(
        self,
        tokens_before: int,
        tokens_after: int,
        messages_before: int,
        messages_after: int,
    ) -> None:
        """Record a compaction event.

        Args:
            tokens_before: Tokens before compaction
            tokens_after: Tokens after compaction
            messages_before: Message count before
            messages_after: Message count after
        """
        self.compaction_events.append({
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_saved": tokens_before - tokens_after,
            "messages_before": messages_before,
            "messages_after": messages_after,
        })

    def __repr__(self) -> str:
        metrics = self.get_metrics()
        return (
            f"RunMiddleware(iterations={metrics['iterations']}, "
            f"tokens={metrics['total_tokens']}, tool_calls={metrics['tool_calls']})"
        )
