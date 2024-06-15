from evaluate import load
from const import cache_dir


class RougeScore:
    def __init__(self, use_stemmer=True):
        self.scorer = load('rouge', cache_dir=cache_dir)
        self.use_stemmer = use_stemmer

    def get_score(self, predictions, references):
        score = self.scorer.compute(predictions=predictions, references=references, use_stemmer=self.use_stemmer,
                                    use_aggregator=False)
        return score
