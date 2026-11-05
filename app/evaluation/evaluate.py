
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.db.session import AsyncSessionLocal
from app.evaluation.retrieval_runner import RetrievalEvaluationRunner
from app.evaluation.generation_runner import GenerationEvaluationRunner


async def run_evaluation():

    print("\nStarting Enterprise Agentic RAG Evaluation")
    print("=" * 60)

    report = {
        "project": "Enterprise Agentic RAG",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    async with AsyncSessionLocal() as session:

        # Stage 1: Retrieval and reranking evaluation.
        print("\n[1/3] Running retrieval and reranking evaluation...")

        retrieval_runner = RetrievalEvaluationRunner(
            session=session,
            top_k=5,
        )

        retrieval_results = await retrieval_runner.run()

        report["retrieval_and_reranking"] = retrieval_results

        print("\nRetrieval and reranking results:")
        print(
            json.dumps(
                retrieval_results["average_metrics"],
                indent=4,
            )
        )

        # Stage 2: Generation evaluation.
        print("\n[2/3] Running generation evaluation...")

        generation_runner = GenerationEvaluationRunner(
            session=session,
            threshold=0.7,
            top_k=5,
        )

        generation_results = await generation_runner.run()

        report["generation"] = generation_results

        print("\nGeneration evaluation results:")
        print(
            json.dumps(
                generation_results["average_scores"],
                indent=4,
            )
        )

        print(
            f"\nPassed metrics: {generation_results['passed_metrics']}"
            f"/{generation_results['total_metrics']}"
        )

        # Stage 3: Overall evaluation summary.
        print("\n[3/3] Preparing consolidated evaluation report...")

    report["overall_status"] = (
        "PASSED"
        if generation_results["overall_passed"]
        else "FAILED"
    )

    report_directory = Path("evaluation_reports")
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        report_directory
        / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    print(f"Overall status: {report['overall_status']}")
    print(f"Report saved at: {report_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
