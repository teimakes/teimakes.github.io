"""Offline adapter for official MuSiQue v1 JSONL; no download or model calls.
Do not pass gold, decomposition, or is_supporting to retrieval/generation.
This adapter prepares the follow-up; the MVP live CLI remains Gaon-only.
"""
import argparse
import json
from pathlib import Path


def adapt(record, row_index):
    paragraphs=[{'id':f"P{p['idx']}",'text':p['title']+'\n'+p['paragraph_text']} for p in record['paragraphs']]
    if len({p['id'] for p in paragraphs})!=len(paragraphs): raise ValueError('Duplicate paragraph indices')
    return {'id':f"{record['id']}:{row_index}", 'official_id':record['id'],
            'input':{'question':record['question'],'paragraphs':paragraphs},
            'gold':{'answers':[record['answer'],*record.get('answer_aliases',[])],
                    'answerable':record['answerable'],
                    'sources':[f"P{p['idx']}" for p in record['paragraphs'] if p['is_supporting']]}}


def export_prediction(example, answer, source_ids, answerable):
    valid={p['id'] for p in example['input']['paragraphs']}
    if not set(source_ids)<=valid: raise ValueError('Invalid support IDs')
    return {'id':example['official_id'],'predicted_answer':answer,
            'predicted_support_idxs':[int(x[1:]) for x in source_ids],
            'predicted_answerable':bool(answerable)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('destination')
    a=p.parse_args()
    with Path(a.destination).open('x',encoding='utf-8') as out:
        for i,line in enumerate(Path(a.source).read_text(encoding='utf-8').splitlines()):
            if line.strip():out.write(json.dumps(adapt(json.loads(line),i),ensure_ascii=False)+'\n')
