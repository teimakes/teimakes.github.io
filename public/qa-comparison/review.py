"""Summarize completed human review without inventing judgments or making API calls."""
import csv
import json
import sys
from collections import defaultdict


def summarize_review(path):
    rows=list(csv.DictReader(open(path,encoding='utf-8-sig')))
    result={}
    for arm in ('full','rag'):
        group=[r for r in rows if r['arm']==arm]
        out={'attempted_rows':len(group)}
        for metric in ('semantic_correct_0_1','faithful_to_context_0_1'):
            values=[r[metric] for r in group if r[metric]!='']
            if any(v not in ('0','1') for v in values):raise ValueError('Use 0, 1 or blank')
            out[metric]={'coverage':len(values)/len(group) if group else 0,
                         'mean':sum(map(int,values))/len(values) if values else None}
        result[arm]=out
    return result

if __name__=='__main__':print(json.dumps(summarize_review(sys.argv[1]),ensure_ascii=False,indent=2))
