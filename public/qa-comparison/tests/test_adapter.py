import unittest
from musique_adapter import adapt,export_prediction
class AdapterTest(unittest.TestCase):
    def test_gold_is_separate(self):
        x=adapt({'id':'2hop__test','question':'where?', 'answer':'SECRET','answer_aliases':[],
                 'answerable':True,'question_decomposition':['SECRET'],
                 'paragraphs':[{'idx':0,'title':'title','paragraph_text':'body','is_supporting':True}]},0)
        self.assertNotIn('SECRET',str(x['input']))
        self.assertNotIn('is_supporting',str(x['input']))
        self.assertEqual(export_prediction(x,'place',['P0'],True)['predicted_support_idxs'],[0])
        with self.assertRaises(ValueError):export_prediction(x,'place',['P99'],True)
    def test_duplicate_official_ids_can_be_kept(self):
        r={'id':'same','question':'q','answer':'a','answerable':False,'paragraphs':[]}
        self.assertNotEqual(adapt(r,0)['id'],adapt(r,1)['id'])
