"""
LLM-based evaluation judge.

This module provides an evaluation judge that uses an LLM to score trajectories.
"""

import json
from typing import Dict, List, Optional

from ..._cancellation_token import CancellationToken
from ...llm import BaseChatCompletionClient
from ...messages import SystemMessage, UserMessage
from ...types import EvalScore, RunTrajectory
from ._base import BaseEvalJudge


class LLMEvalJudge(BaseEvalJudge):
    """LLM-based evaluation judge that uses another model to score trajectories."""

    def __init__(
        self,
        client: BaseChatCompletionClient,
        name: Optional[str] = None,
        default_criteria: Optional[List[str]] = None,
        answer_strategy: str = "last_non_empty",
        custom_instructions: Optional[str] = None,
    ):
        """Initialize the LLM judge.

        Args:
            client: LLM client to use for scoring
            name: Optional custom name (defaults to model name)
            default_criteria: Default evaluation criteria if none specified
            answer_strategy: How to extract answer from trajectory
                Note: LLM judges often benefit from seeing full context
            custom_instructions: Optional additional instructions to append to the system prompt
                Use this to add domain-specific guidance, adjust for multi-agent evaluation,
                or specify format flexibility requirements
        """
        super().__init__(
            name or f"LLM-{getattr(client, 'model', 'Judge')}", answer_strategy
        )
        self.client = client
        self.default_criteria = default_criteria or [
            "accuracy",
            "completeness",
            "helpfulness",
        ]
        self.custom_instructions = custom_instructions

    async def score(
        self,
        trajectory: RunTrajectory,
        criteria: Optional[List[str]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> EvalScore:
        """Score an evaluation trajectory using LLM.

        Args:
            trajectory: The execution trajectory to score
            criteria: Optional list of evaluation dimensions to score
            cancellation_token: Optional token to cancel scoring

        Returns:
            EvalScore with overall and dimensional scores
        """
        # Use provided criteria, task-level criteria, or defaults
        eval_criteria = criteria or trajectory.task.eval_criteria or self.default_criteria

        # Read rubric directly from task (first-class field)
        rubric = trajectory.task.rubric

        try:
            # Build the evaluation prompt
            system_prompt = self._build_system_prompt(eval_criteria, rubric)
            user_prompt = self._build_user_prompt(trajectory)

            messages = [
                SystemMessage(content=system_prompt, source="system"),
                UserMessage(content=user_prompt, source="user"),
            ]

            # Get LLM response (note: cancellation_token not passed to client as it's not part of the base interface)
            result = await self.client.create(messages)
            response_content = result.message.content

            # Parse the structured response
            score_data = self._parse_llm_response(response_content, eval_criteria)

            return EvalScore(
                overall=score_data["overall"],
                dimensions=score_data["dimensions"],
                reasoning=score_data["reasoning"],
                trajectory=trajectory,
                metadata={
                    "judge_name": self.name,
                    "model": result.model,
                    "criteria_used": eval_criteria,
                    "raw_response": response_content,
                },
            )

        except Exception as e:
            # Return fallback score on error
            return EvalScore(
                overall=5.0,  # Neutral score
                dimensions={dim: 5.0 for dim in eval_criteria},
                reasoning={dim: f"Judge error: {str(e)}" for dim in eval_criteria},
                trajectory=trajectory,
                metadata={
                    "judge_name": self.name,
                    "error": str(e),
                    "criteria_used": eval_criteria,
                },
            )

    def _build_system_prompt(self, criteria: List[str], rubric: Optional[Dict[str, str]] = None) -> str:
        """Build the system prompt for the evaluation LLM.

        Args:
            criteria: List of evaluation dimensions to score
            rubric: Optional rubric from the task - maps criterion name to
                scoring guidance (e.g. "10: All files reviewed. 5: Most. 0: None.")
                When provided, rubric descriptions replace the generic defaults.
        """
        default_descriptions = {
            "accuracy": "How factually correct and truthful is the response?",
            "completeness": "How thoroughly does the response address the task?",
            "helpfulness": "How useful and actionable is the response?",
            "clarity": "How clear and well-structured is the response?",
            "creativity": "How creative and original is the response?",
            "safety": "How safe and appropriate is the response?",
        }

        criteria_details = []
        for criterion in criteria:
            # Rubric takes priority over generic defaults
            if rubric and criterion in rubric:
                description = rubric[criterion]
            else:
                description = default_descriptions.get(
                    criterion, f"Quality of {criterion}"
                )
            criteria_details.append(f"- {criterion}: {description}")

        base_prompt = f"""You are an expert evaluation judge. Your task is to score AI agent conversations based on specific criteria.

Evaluation Criteria (each scored 0-10):
{chr(10).join(criteria_details)}

Instructions:
1. Analyze the task, expected output (if provided), and the complete agent conversation
2. Consider both the final outcome AND the process (reasoning, communication, error handling)
3. Score each criterion from 0-10 (0=poor, 5=average, 10=excellent)
4. Calculate overall score as average of dimensional scores
5. Provide brief reasoning for each score"""

        # Append custom instructions if provided
        if self.custom_instructions:
            base_prompt += f"\n\nAdditional Evaluation Guidance:\n{self.custom_instructions}"

        base_prompt += f"""

Respond with this EXACT JSON format:
{{
  "overall": <overall_score>,
  "dimensions": {{
    "{criteria[0]}": <score>,
    {', '.join(f'"{c}": <score>' for c in criteria[1:]) if len(criteria) > 1 else ''}
  }},
  "reasoning": {{
    "{criteria[0]}": "<brief_reason>",
    {', '.join(f'"{c}": "<brief_reason>"' for c in criteria[1:]) if len(criteria) > 1 else ''}
  }}
}}"""

        return base_prompt

    def _build_user_prompt(self, trajectory: RunTrajectory) -> str:
        """Build the user prompt containing the trajectory to evaluate.

        Formats each message with explicit role labels and full tool call
        details so the judge can see exactly what happened.
        """
        task_info = f"Task: {trajectory.task.name}\nInput: {trajectory.task.input}"

        if trajectory.task.expected_output:
            task_info += f"\nExpected Output: {trajectory.task.expected_output}"

        if trajectory.success and trajectory.messages:
            formatted_msgs = []
            for msg in trajectory.messages:
                formatted_msgs.append(self._format_message(msg))
            actual_output = "\n\n".join(formatted_msgs)

            conversation_summary = f"Messages exchanged: {len(trajectory.messages)}"
            if trajectory.usage:
                conversation_summary += f", Tokens: {trajectory.usage.tokens_input + trajectory.usage.tokens_output}"
        else:
            actual_output = f"EXECUTION FAILED: {trajectory.error or 'Unknown error'}"
            conversation_summary = "No successful execution"

        return f"""{task_info}

Execution Summary: {conversation_summary}
Success: {trajectory.success}

Complete Agent Conversation:
{actual_output}

Please evaluate this complete conversation according to the specified criteria."""

    def _format_message(self, msg) -> str:
        """Format a single message with role label and full structure.

        Shows tool calls, tool results, and proper role labels so the
        judge sees exactly what happened in the conversation.
        """
        role = getattr(msg, "role", "unknown")
        source = getattr(msg, "source", "")
        content = getattr(msg, "content", "")

        if role == "system":
            return f"[SYSTEM ({source})]\n{content}"

        elif role == "user":
            return f"[USER ({source})]\n{content}"

        elif role == "assistant":
            parts = [f"[ASSISTANT ({source})]"]
            if content:
                parts.append(content)

            # Show tool calls with names and parameters
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    tool_name = getattr(tc, "tool_name", str(tc))
                    params = getattr(tc, "parameters", {})
                    # Truncate large parameter values for readability
                    param_strs = []
                    for k, v in params.items():
                        v_str = str(v)
                        if len(v_str) > 200:
                            v_str = v_str[:200] + "..."
                        param_strs.append(f"  {k}: {v_str}")
                    param_block = "\n".join(param_strs)
                    parts.append(f"[TOOL CALL: {tool_name}]\n{param_block}")

            return "\n".join(parts)

        elif role == "tool":
            tool_name = getattr(msg, "tool_name", "unknown")
            success = getattr(msg, "success", True)
            error = getattr(msg, "error", None)
            status = "SUCCESS" if success else "FAILED"
            header = f"[TOOL RESULT ({tool_name}) - {status}]"
            if error:
                header += f"\nError: {error}"
            # Truncate very long tool output for judge readability
            display_content = content
            if len(display_content) > 2000:
                display_content = display_content[:2000] + "\n... (truncated)"
            return f"{header}\n{display_content}"

        else:
            return f"[{role.upper()} ({source})]\n{content}"

    def _parse_llm_response(self, response: str, criteria: List[str]) -> Dict:
        """Parse the LLM's structured response."""
        try:
            # Try to extract JSON from the response
            response = response.strip()

            # Find JSON block if wrapped in code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()

            # Parse the JSON
            parsed = json.loads(response)

            # Validate structure
            if not all(key in parsed for key in ["overall", "dimensions", "reasoning"]):
                raise ValueError("Missing required keys")

            # Ensure all criteria are present
            for criterion in criteria:
                if criterion not in parsed["dimensions"]:
                    parsed["dimensions"][criterion] = 5.0
                if criterion not in parsed["reasoning"]:
                    parsed["reasoning"][criterion] = "No reasoning provided"

            return parsed

        except Exception:
            # Fallback to neutral scores
            return {
                "overall": 5.0,
                "dimensions": {dim: 5.0 for dim in criteria},
                "reasoning": {
                    dim: "Failed to parse judge response" for dim in criteria
                },
            }
