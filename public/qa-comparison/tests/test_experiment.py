import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import experiment as e
from retrieval import BM25,chunks
from scoring import Answer,check_fields

class Tests(unittest.TestCase):
    def setUp(self):
        self.config=e.read(e.ROOT/'config.json')
        self.case=e.load_cases('dev')[0]
        self.doc=(e.ROOT/'data/handbook.txt').read_text(encoding='utf-8')
    def test_reused_dataset_integrity(self):
        a,b=e.load_cases('dev'),e.load_cases('holdout')
        self.assertEqual((len(a),len(b)),(12,12))
        self.assertFalse(set(x['id'] for x in a)&set(x['id'] for x in b))
        ids={p['id'] for p in chunks(self.doc)}
        self.assertEqual(len(ids),31)
        for c in a+b:self.assertTrue(set(c['sources'])<=ids)
    def test_numeric_wrong_is_wrong(self):
        a=Answer(answer='wrong',verdict='value',value=153000,sources=['A02'])
        self.assertEqual(check_fields(a,self.case,{'A02'})['structured_match'],0)
    def test_no_gold_leakage(self):
        case=copy.deepcopy(self.case);case['reference_answer']='SECRET_GOLD';case['sources']=['SECRET_SOURCE']
        with patch.object(e,'load_cases',return_value=[case]):
            work=list(e.jobs(self.config,'dev',0,1))
        for j in work:
            self.assertNotIn('SECRET',e.dump(j['request']))
            self.assertNotIn('expected',e.dump(j['request']))
    def test_identical_instructions_and_question(self):
        a,b=list(e.jobs(self.config,'dev',1,1))
        self.assertEqual(a['request']['input'][0],b['request']['input'][0])
        self.assertEqual(a['case']['question'],b['case']['question'])
        self.assertEqual(len(b['provided_ids']),5)
        self.assertEqual(len(a['provided_ids']),31)
    def test_retrieval_deterministic_and_no_gold(self):
        idx=BM25(chunks(self.doc));a=idx.search('G01 정가',5)
        self.assertEqual(a,idx.search('G01 정가',5))
        self.assertIn('A02',[x['id'] for x in a])
    def test_invalid_citation_penalized(self):
        a=Answer(answer='x',verdict='value',value=120000,sources=['A02','FAKE'])
        m=e.evidence(a,self.case,['A02'])
        self.assertEqual(m['citation_precision'],.5);self.assertEqual(m['citations_in_context'],0)
    def test_plan_free_and_approval_gate(self):
        with patch('urllib.request.urlopen',side_effect=AssertionError('NETWORK')) as net:
            p=e.plan(self.config);self.assertEqual(p['max_calls'],4)
            with self.assertRaises(ValueError):e.verify_plan(p,'')
            e.verify_plan(p,p['plan_sha256']);net.assert_not_called()
    def test_modified_plan_rejected(self):
        p=e.plan(self.config);p['config']['max_output_tokens']=999
        with self.assertRaises(ValueError):e.verify_plan(p,p['plan_sha256'])
    def test_invalid_config(self):
        for v in (float('nan'),-1,0):
            c=dict(self.config,project_cap_usd=v)
            with self.assertRaises(ValueError):e.plan(c)
    def test_account_gate(self):
        p=e.plan(self.config)
        with self.assertRaises(ValueError):e.account_gate(p,{})
        with self.assertRaises(ValueError):e.account_gate(p,{},datetime(2026,11,8,tzinfo=timezone.utc))
    def test_ledger_reserves_and_prevents_replay(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ledger.json'
            with e.Ledger(path,.25) as ledger:
                ledger.reserve('p','a',.2)
                with self.assertRaises(ValueError):ledger.reserve('p','b',.1)
                with self.assertRaises(ValueError):ledger.reserve('p','a',.01)
            with e.Ledger(path,.25) as ledger:
                self.assertAlmostEqual(ledger.data['reserved_usd'],.2)
    def test_lock_excludes_concurrent_run(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ledger.json'
            with e.Ledger(path,.25):
                with self.assertRaises(FileExistsError):
                    with e.Ledger(path,.25):pass
    def test_usage_includes_reasoning_once(self):
        m=e.usage_cost({'input_tokens':100,'output_tokens':20,'input_tokens_details':{'cached_tokens':50},'output_tokens_details':{'reasoning_tokens':10}},self.config)
        self.assertAlmostEqual(m['token_cost_estimate_usd'],.0000155)
        self.assertEqual(m['reasoning_tokens'],10)
    def test_incomplete_fails(self):
        with self.assertRaises(ValueError):e.parse_response({'status':'incomplete','output':[]})
    def test_response_schema(self):
        for v in [True,float('inf')]:
            with self.assertRaises(ValueError):Answer(answer='x',verdict='value',value=v,sources=[])
    def test_live_mock_end_to_end(self):
        p=e.plan(self.config,limit=2)
        response={'id':'mock-only','model':'mock-not-real','status':'completed',
                  'usage':{'input_tokens':100,'output_tokens':20},'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps({'answer':'고정 테스트 응답','verdict':'value','value':120000,'sources':['A02']})}]}]}
        with tempfile.TemporaryDirectory() as d, patch.object(e,'account_gate'),patch.object(e,'complete',return_value=(response,.1)) as api:
            s=e.run(p,p['plan_sha256'],{},Path(d)/'out',Path(d)/'ledger.json')
            self.assertEqual(api.call_count,4)
            self.assertEqual(s['arms']['full']['structured_match_rate'],.5)
            self.assertTrue((Path(d)/'out/human_review.csv').exists())
    def test_failure_stops_and_keeps_reservation(self):
        p=e.plan(self.config)
        with tempfile.TemporaryDirectory() as d,patch.object(e,'account_gate'),patch.object(e,'complete',side_effect=TimeoutError) as api:
            s=e.run(p,p['plan_sha256'],{},Path(d)/'out',Path(d)/'ledger.json')
            self.assertEqual(api.call_count,1)
            self.assertEqual(s['arms']['full']['structured_match_rate'],0)
            self.assertGreater(e.read(Path(d)/'ledger.json')['reserved_usd'],0)
    def test_full_dev_plan_under_cap(self):
        p=e.plan(self.config,limit=0)
        self.assertEqual(p['max_calls'],24)
        self.assertLess(p['reserved_usd'],self.config['project_cap_usd'])

if __name__=='__main__':unittest.main()
