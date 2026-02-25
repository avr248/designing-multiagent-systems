"""
Basic evaluation example for picoagents.

This example demonstrates how to evaluate different components using the new evaluation system.
"""

import asyncio
import os

from picoagents import Agent, OpenAIChatCompletionClient
from picoagents.eval import AgentEvalTarget, EvalRunner, LLMEvalJudge, ModelEvalTarget
from picoagents.types import Task


async def main():
    """Run a basic evaluation example."""
    print("🧪 Basic Picoagents Evaluation Example")
    print("=" * 50)

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    # Create LLM client
    client = OpenAIChatCompletionClient(model="gpt-4.1-mini", api_key=api_key)

    # Create evaluation tasks
    tasks = [
        Task(name="Math Problem", input="What is 25 * 16?", expected_output="400"),
        Task(
            name="Capital City",
            input="What is the capital of France?",
            expected_output="Paris",
        ),
        Task(
            name="Creative Writing",
            input="Write a haiku about programming.",
            expected_output="A creative haiku about programming",
        ),
    ]

    print(f"📝 Created {len(tasks)} evaluation tasks")

    # Create evaluation targets

    # 1. Direct model target
    model_target = ModelEvalTarget(
        client=client,
        name="gpt-4.1-nano",
        system_message="You are a helpful assistant. Give clear, concise answers.",
    )

    # 2. Agent target
    agent = Agent(
        name="test-agent",
        description="A helpful test agent for evaluation",
        instructions="You are a helpful assistant. Provide accurate and helpful responses.",
        model_client=client,
    )
    agent_target = AgentEvalTarget(agent, name="gpt-4.1-nano-Agent")

    print("🎯 Created evaluation targets:")
    print(f"  - {model_target.name}")
    print(f"  - {agent_target.name}")

    # Create LLM judge
    judge_client = OpenAIChatCompletionClient(model="gpt-4.1-mini", api_key=api_key)
    judge = LLMEvalJudge(
        client=judge_client,
        name="gpt-4.1-mini-Judge",
        default_criteria=["accuracy", "helpfulness", "clarity"],
    )

    print(f"👨‍⚖️  Created judge: {judge.name}")

    # Create evaluation runner
    runner = EvalRunner(judge=judge, parallel_tasks=True)

    print(f"🏃 Created runner with parallel execution")

    # Run evaluations
    print("\n🚀 Running Evaluations...")
    print("-" * 30)

    # Evaluate model target
    print(f"\n📊 Evaluating {model_target.name}...")
    model_scores = await runner.evaluate(model_target, tasks)

    # Evaluate agent target
    print(f"📊 Evaluating {agent_target.name}...")
    agent_scores = await runner.evaluate(agent_target, tasks)

    # Display results
    print("\n📈 RESULTS")
    print("=" * 50)

    targets = [("Direct Model", model_scores), ("Agent", agent_scores)]

    for target_name, scores in targets:
        print(f"\n{target_name} Results:")
        print("-" * 25)

        total_score = 0
        for i, (task, score) in enumerate(zip(tasks, scores)):
            print(f"{i+1}. {task.name}")
            print(f"   Overall Score: {score.overall:.1f}/10")
            print(
                f"   Dimensions: {', '.join(f'{k}={v:.1f}' for k, v in score.dimensions.items())}"
            )

            # Show one reasoning example
            first_reasoning = (
                list(score.reasoning.values())[0] if score.reasoning else "No reasoning"
            )
            if len(first_reasoning) > 60:
                first_reasoning = first_reasoning[:60] + "..."
            print(f"   Reasoning: {first_reasoning}")

            # Show the actual response that was judged
            actual_response = score.get_final_response()
            if len(actual_response) > 100:
                actual_response = actual_response[:100] + "..."
            print(f"   Actual Response: {actual_response}")
            print()

            total_score += score.overall

        avg_score = total_score / len(scores) if scores else 0
        print(f"Average Score: {avg_score:.1f}/10")

    # Compare targets
    print(f"\n🏆 COMPARISON")
    print("=" * 30)

    if len(targets) >= 2:
        model_avg = sum(s.overall for s in model_scores) / len(model_scores)
        agent_avg = sum(s.overall for s in agent_scores) / len(agent_scores)

        if model_avg > agent_avg:
            winner = "Direct Model"
            diff = model_avg - agent_avg
        elif agent_avg > model_avg:
            winner = "Agent"
            diff = agent_avg - model_avg
        else:
            winner = "Tie"
            diff = 0

        if winner != "Tie":
            print(f"Winner: {winner} (+{diff:.1f} points)")
        else:
            print("Result: Tie!")

        print(f"Direct Model Avg: {model_avg:.1f}")
        print(f"Agent Avg: {agent_avg:.1f}")

    print(f"\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
