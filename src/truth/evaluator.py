class TruthEvaluator:
    def evaluate(self, candidate: str, ground_truth: str) -> bool:
        return candidate.lower() == ground_truth.lower()
