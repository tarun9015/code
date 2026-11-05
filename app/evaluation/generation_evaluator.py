
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase


class GenerationEvaluator:

    def __init__(self, threshold: float = 0.7):

        self.metrics = [
            FaithfulnessMetric(
                threshold=threshold,
                model="gpt-4o-mini",
            ),
            AnswerRelevancyMetric(
                threshold=threshold,
                model="gpt-4o-mini",
            ),
            ContextualPrecisionMetric(
                threshold=threshold,
                model="gpt-4o-mini",
            ),
            ContextualRecallMetric(
                threshold=threshold,
                model="gpt-4o-mini",
            ),
        ]

    def evaluate(
        self,
        question: str,
        actual_answer: str,
        expected_answer: str,
        retrieved_contexts: list[str],
    ) -> dict:

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer,
            retrieval_context=retrieved_contexts,
        )

        results = {}

        for metric in self.metrics:
            metric.measure(test_case)

            results[metric.__class__.__name__] = {
                "score": metric.score,
                "passed": metric.is_successful(),
                "reason": metric.reason,
            }

        return results
