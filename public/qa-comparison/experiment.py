"""Plan offline; explicit approval + verified account gate before any HTTP request.
Python 3.11+. No SDK retries, embeddings, judge, telemetry or hosted storage.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time
import urllib.request
import urllib.error
from prompts import messages
from scoring import Answer, check_fields
from retrieval import BM25, chunks

ROOT = Path(__file__).resolve().parent
ENDPOINT = 'https://api.openai.com/v1/responses'
SCHEMA = {'type':'object', 'additionalProperties':False, 'properties': {
    'answer':{'type':'string'}, 'verdict':{'type':'string','enum':['yes','no','value','unknown']},
    'value':{'type':['number','null']}, 'sources':{'type':'array','items':{'type':'string'}}},
    'required':['answer','verdict','value','sources']}


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False)


def sha(obj):
    return hashlib.sha256(dump(obj).encode()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_cases(split):
    return [json.loads(x) for x in (ROOT/f'data/{split}.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]


def validate_config(c):
    for k in ('max_output_tokens','top_k'):
        if type(c[k]) is not int or c[k] <= 0:
            raise ValueError(f'Invalid {k}')
    for k in ('input_usd_per_million','cached_usd_per_million','output_usd_per_million',
              'input_reserve_usd_per_million','project_cap_usd'):
        if type(c[k]) not in (float,int) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(f'Invalid {k}')
    if c['input_reserve_usd_per_million'] < c['input_usd_per_million']:
        raise ValueError('Reserve input rate must cover regular input rate')


def payload(c, document, question):
    return {'model':c['model'], 'input':messages(document, question, 'grounded'),
            'reasoning':{'effort':c['reasoning_effort']}, 'max_output_tokens':c['max_output_tokens'],
            'store':False, 'service_tier':'default',
            'text':{'format':{'type':'json_schema','name':'qa_answer','strict':True,'schema':SCHEMA}}}


def jobs(c, split, limit, repeats):
    document = (ROOT/'data/handbook.txt').read_text(encoding='utf-8')
    paragraphs = chunks(document)
    index_start = time.perf_counter()
    index = BM25(paragraphs)
    index_s = time.perf_counter()-index_start
    cases = load_cases(split)
    if limit: cases = cases[:limit]
    for repeat in range(repeats):
        for i, case in enumerate(cases):
            # Retrieval sees only the question; gold IDs/answers are not arguments.
            start = time.perf_counter()
            selected = index.search(case['question'], c['top_k'])
            retrieval_s = time.perf_counter()-start
            arms = ['full','rag'] if (i+repeat)%2 == 0 else ['rag','full']
            for arm in arms:
                context = document if arm == 'full' else '\n\n'.join(p['text'] for p in selected)
                yield {'key':f"{case['id']}:{repeat}:{arm}", 'arm':arm, 'repeat':repeat,
                       'case':case, 'provided_ids':[p['id'] for p in (paragraphs if arm=='full' else selected)],
                       'retrieval_s':retrieval_s if arm=='rag' else 0.0,
                       'index_build_s':index_s if arm=='rag' else 0.0,
                       'request':payload(c, context, case['question'])}


def input_bound(request):
    # Deliberately pessimistic: UTF-8 bytes of complete JSON plus 1024 overhead.
    # Not a tokenizer count or a provider-enforced input limit.
    return len(dump(request).encode('utf-8'))+1024


def reserve(request, c):
    n = input_bound(request)
    if n > 100000:
        raise ValueError('MVP safety limit: input bound >100k; re-plan long-context experiment')
    return (n*c['input_reserve_usd_per_million'] + c['max_output_tokens']*c['output_usd_per_million'])/1e6


def fingerprints():
    files = sorted([*ROOT.glob('*.py'), *ROOT.glob('data/*.txt'), *ROOT.glob('data/*.jsonl')])
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def plan(c, split='dev', limit=2, repeats=1):
    validate_config(c)
    if limit < 0 or repeats < 1: raise ValueError('limit >=0, repeats >=1')
    work = list(jobs(c,split,limit,repeats))
    if not work: raise ValueError('Empty dataset')
    calls = [{'key':j['key'],'request_sha256':sha(j['request']),
              'input_byte_bound':input_bound(j['request']),'reserved_usd':reserve(j['request'],c)} for j in work]
    obj = {'version':1,'config':c,'split':split,'limit':limit,'repeats':repeats,
           'calls':calls,'max_calls':len(calls),'reserved_usd':sum(x['reserved_usd'] for x in calls),
           'files':fingerprints(), 'status':'AWAITING_APPROVAL_NO_LIVE_RESULTS'}
    obj['plan_sha256'] = sha(obj)
    return obj


def verify_plan(p, approval):
    if approval != p['plan_sha256']: raise ValueError('Explicit approval hash required')
    actual = plan(p['config'],p['split'],p['limit'],p['repeats'])
    if actual != p: raise ValueError('Plan/data/code changed; regenerate and obtain approval again')
    if p['reserved_usd'] > p['config']['project_cap_usd']: raise ValueError('Plan exceeds project cap')


def account_gate(p, account, now=None):
    now = now or datetime.now(timezone.utc)
    c = p['config']
    if now >= datetime.fromisoformat(c['stop_before_utc']): raise ValueError('Credit expiry safety cutoff reached')
    required = ('organization_id','project_id','checked_at_utc','remaining_credit_usd',
                'model_access_verified','credit_eligible_verified')
    if any(account.get(k) is None for k in required): raise ValueError('Account verification incomplete')
    age = (now-datetime.fromisoformat(account['checked_at_utc'])).total_seconds()
    if age < 0 or age > 86400: raise ValueError('Recheck account within 24h')
    if account['model_access_verified'] is not True or account['credit_eligible_verified'] is not True:
        raise ValueError('Model access and promotional eligibility must be verified')
    balance = account['remaining_credit_usd']
    if type(balance) not in (int,float) or not math.isfinite(balance) or balance < c['project_cap_usd']:
        raise ValueError('Verified balance must cover project cap')
    if not os.environ.get('OPENAI_API_KEY'): raise ValueError('No securely configured OPENAI_API_KEY')
    if account['organization_id'] != os.environ.get('OPENAI_ORG_ID') or account['project_id'] != os.environ.get('OPENAI_PROJECT_ID'):
        raise ValueError('Organization/project must match verified account')


class Ledger:
    """Exclusive lock across runs; reserve before send, never refund uncertain charges.
    Keep this single file across every live run. It is a local guard, not an account cap.
    """
    def __init__(self,path,cap): self.path,self.cap=Path(path),cap
    def __enter__(self):
        self.lock = self.path.with_suffix('.lock')
        self.fd = os.open(self.lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        self.data=read(self.path) if self.path.exists() else {'reserved_usd':0,'calls':[]}
        return self
    def reserve(self, plan_hash, key, usd):
        identity=plan_hash+':'+key
        if any(x['identity']==identity for x in self.data['calls']): raise ValueError('Call already attempted; no automatic replay')
        if self.data['reserved_usd']+usd > self.cap+1e-12: raise ValueError('Project budget exhausted')
        self.data['reserved_usd']+=usd
        self.data['calls'].append({'identity':identity,'reserved_usd':usd})
        temp=self.path.with_suffix('.tmp')
        with temp.open('w',encoding='utf-8') as f:
            f.write(dump(self.data)); f.flush(); os.fsync(f.fileno())
        os.replace(temp,self.path)
    def __exit__(self,*args):
        os.close(self.fd); self.lock.unlink()


def complete(request):
    headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY'], 'Content-Type':'application/json',
             'OpenAI-Organization':os.environ['OPENAI_ORG_ID'],'OpenAI-Project':os.environ['OPENAI_PROJECT_ID']}
    req=urllib.request.Request(ENDPOINT,dump(request).encode(),headers=headers,method='POST')
    start=time.perf_counter()
    try:
        with urllib.request.urlopen(req,timeout=120) as response:
            data=json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}; stopped; no retry') from None
    except (urllib.error.URLError,TimeoutError):
        raise RuntimeError('Network/timeout; billing uncertain; stopped; no retry') from None
    return data,time.perf_counter()-start


def parse_response(data):
    if data.get('status') != 'completed': raise ValueError('Incomplete/refused response')
    text=''.join(c['text'] for o in data.get('output',[]) if o.get('type')=='message'
                 for c in o.get('content',[]) if c.get('type')=='output_text')
    return Answer.model_validate_json(text),text


def usage_cost(usage,c):
    i,o=usage['input_tokens'],usage['output_tokens']
    cached=usage.get('input_tokens_details',{}).get('cached_tokens',0)
    if any(type(n) is not int or n<0 for n in (i,o,cached)) or cached>i: raise ValueError('Invalid usage')
    return {'input_tokens':i,'cached_input_tokens':cached,'output_tokens':o,
            'reasoning_tokens':usage.get('output_tokens_details',{}).get('reasoning_tokens',0),
            'token_cost_estimate_usd':((i-cached)*c['input_usd_per_million']+cached*c['cached_usd_per_million']+o*c['output_usd_per_million'])/1e6,
            'token_cost_conservative_usd':(i*c['input_reserve_usd_per_million']+o*c['output_usd_per_million'])/1e6}


def evidence(answer, case, provided):
    cited,gold=set(answer.sources),set(case['sources'])
    hit=len(cited&gold)
    precision=hit/len(cited) if cited else 0.0
    recall=hit/len(gold) if gold else None
    return {'citation_precision':precision,'citation_recall':recall,
            'citation_f1':2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.0,
            'citations_in_context':int(bool(cited) and cited<=set(provided)),
            'retrieval_gold_recall':len(gold&set(provided))/len(gold) if gold else None}


def summarize(rows, expected):
    result={}
    for arm in ('full','rag'):
        part=[r for r in rows if r['arm']==arm]
        n=expected//2
        summary={'planned':n,'attempted':len(part),'success':sum(r['status']=='ok' for r in part),
                 'structured_match_rate':sum(r.get('structured_match',0) for r in part)/n,
                 'citation_f1_mean_all_planned':sum(r.get('citation_f1',0) for r in part)/n,
                 'unattempted':n-len(part),'semantic_faithfulness':'PENDING_HUMAN_REVIEW'}
        for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_tokens','token_cost_estimate_usd','token_cost_conservative_usd'):
            summary[k]=sum(r.get(k,0) for r in part)
        summary['usage_coverage']=sum('input_tokens' in r for r in part)/n
        for metric in ('api_latency_s','retrieval_s','e2e_s'):
            vals=sorted(r[metric] for r in part if metric in r)
            summary[metric+'_p50']=statistics.median(vals) if vals else None
            summary[metric+'_p95']=vals[max(0,math.ceil(len(vals)*.95)-1)] if vals else None
        result[arm]=summary
    paired=[]
    for full in [r for r in rows if r['arm']=='full']:
        rag=next((r for r in rows if r['arm']=='rag' and r['id']==full['id'] and r['repeat']==full['repeat']),None)
        if rag:
            paired.append({'id':full['id'],'repeat':full['repeat'],
                           'full_correct':full.get('structured_match',0),'rag_correct':rag.get('structured_match',0),
                           'rag_minus_full_tokens':rag['input_tokens']-full['input_tokens'] if 'input_tokens' in rag and 'input_tokens' in full else None})
    return {'arms':result,'paired':paired,'warning':'Costs are token estimates, not billing balance. Small single-document pilot.'}


def run(p, approval, account, out, ledger_path=None):
    verify_plan(p,approval)
    account_gate(p,account)
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',dict(p,created_utc=datetime.now(timezone.utc).isoformat(),
                                 pydantic=importlib.metadata.version('pydantic'),data_status='LIVE_API'))
    rows=[]
    with Ledger(ledger_path or ROOT/'budget-ledger.json',p['config']['project_cap_usd']) as ledger:
        for job, planned in zip(jobs(p['config'],p['split'],p['limit'],p['repeats']),p['calls']):
            account_gate(p,account)
            ledger.reserve(p['plan_sha256'],job['key'],planned['reserved_usd'])
            case=job['case']
            row={'id':case['id'],'arm':job['arm'],'repeat':job['repeat'],'question':case['question'],
                 'expected':case['expected'],'reference_answer':case['reference_answer'],
                 'reference_sources':case['sources'],'provided_ids':job['provided_ids'],
                 'retrieval_s':job['retrieval_s'],'index_build_s':job['index_build_s'],
                 'reserved_usd':planned['reserved_usd'],'status':'error','semantic_faithfulness':None}
            row['request']=job['request']
            started=time.perf_counter()
            try:
                data,latency=complete(job['request'])
                row.update(raw_response=data,api_latency_s=latency,resolved_model=data.get('model'),response_id=data.get('id'))
                row.update(usage_cost(data['usage'],p['config']))
                if row['input_tokens']>planned['input_byte_bound'] or row['output_tokens']>p['config']['max_output_tokens']:
                    raise ValueError('Usage exceeded reservation assumptions; stop')
                answer,text=parse_response(data)
                row.update(parsed=answer.model_dump(),text=text)
                all_ids=set(p['id'] for p in chunks((ROOT/'data/handbook.txt').read_text(encoding='utf-8')))
                row.update(check_fields(answer,case,all_ids),**evidence(answer,case,job['provided_ids']))
                row['status']='ok'
            except Exception as e:
                row['error_type']=type(e).__name__
                # Do not include HTTP body, headers, credentials or arbitrary exception text.
            row['e2e_s']=time.perf_counter()-started+job['retrieval_s']
            rows.append(row)
            with (out/'results.jsonl').open('a',encoding='utf-8') as f:
                f.write(dump(row)+'\n'); f.flush(); os.fsync(f.fileno())
            write(out/'summary.json',summarize(rows,p['max_calls']))
            if row['status']!='ok': break
    review=[{'id':r['id'],'arm':r['arm'],'repeat':r['repeat'],'answer':r.get('text',''),
             'gold_answer':r['reference_answer'],'gold_sources':','.join(r['reference_sources']),
             'semantic_correct_0_1':'','faithful_to_context_0_1':'','notes':''} for r in rows]
    with (out/'human_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(review[0]) if review else ['id']); w.writeheader();w.writerows(review)
    summary=summarize(rows,p['max_calls'])
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['arm',*summary['arms']['full']]);w.writeheader()
        for arm,values in summary['arms'].items():w.writerow(dict(arm=arm,**values))
    return summarize(rows,p['max_calls'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('plan'); q.add_argument('--split',choices=['dev','holdout'],default='dev')
    q.add_argument('--limit',type=int,default=2);q.add_argument('--repeats',type=int,default=1)
    q.add_argument('--out',required=True);q.add_argument('--config',default=str(ROOT/'config.json'))
    r=sub.add_parser('run');r.add_argument('--plan',required=True);r.add_argument('--approve',required=True)
    r.add_argument('--account',required=True);r.add_argument('--out',required=True)
    a=parser.parse_args()
    if a.command=='plan':
        p=plan(read(a.config),a.split,a.limit,a.repeats);write(a.out,p)
        print(dump({'plan_sha256':p['plan_sha256'],'calls':p['max_calls'],'reserved_usd':p['reserved_usd'],'live_calls':0}))
    else: print(json.dumps(run(read(a.plan),a.approve,read(a.account),a.out),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
