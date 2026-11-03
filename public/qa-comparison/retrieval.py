"""Deterministic local BM25; Korean word + within-word character bigrams."""
import math
import re
from collections import Counter


def chunks(document):
    starts = list(re.finditer(r'^([ABC]\d{2}) ', document, re.M))
    if not starts:
        raise ValueError('No paragraph IDs found')
    return [{'id': m.group(1), 'text': document[m.start():starts[i+1].start() if i+1 < len(starts) else len(document)].strip()}
            for i, m in enumerate(starts)]


def tokens(text):
    words = re.findall(r'[a-z0-9]+|[가-힣]+', text.lower())
    return words + [w[i:i+2] for w in words if re.fullmatch('[가-힣]+', w) for i in range(len(w)-1)]


class BM25:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs
        self.counts = [Counter(tokens(p['text'])) for p in paragraphs]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg = sum(self.lengths)/len(self.lengths)
        self.df = Counter(t for c in self.counts for t in c)

    def search(self, question, k=5):
        query = set(tokens(question))
        scores = []
        for i, c in enumerate(self.counts):
            score = 0.0
            for t in query:
                f = c[t]
                if f:
                    idf = math.log(1 + (len(self.counts)-self.df[t]+0.5)/(self.df[t]+0.5))
                    score += idf*f*2.5/(f+1.5*(0.25+0.75*self.lengths[i]/self.avg))
            scores.append((i, score))
        chosen = sorted(scores, key=lambda x: (-x[1], x[0]))[:k]
        # Present retrieved paragraphs in original document order in both arms.
        return [dict(self.paragraphs[i], retrieval_score=s) for i, s in sorted(chosen)]
