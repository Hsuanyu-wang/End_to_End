from deepeval.scorer import Scorer
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type="rougeL" #rougeL, rouge2, rouge1
        )
        self.success = self.score >= self.threshold
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"
    
class BLEUMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.sentence_bleu_score(
            prediction=test_case.actual_output,
            references=test_case.expected_output,
            bleu_type="bleu1" #bleu1, bleu2, bleu3, bleu4
        )
        self.success = self.score >= self.threshold
        return self.score
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "BLEU Metric"
    
    
class F1Metric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.f1_score(
            targets=test_case.expected_output,
            prediction=test_case.actual_output,
            f1_type="f1"
        )
        self.success = self.score >= self.threshold
        return self.score
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "F1 Metric"

