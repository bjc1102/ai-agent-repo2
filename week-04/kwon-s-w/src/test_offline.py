"""Offline checks only. Fake API responses never enter real experiment outputs."""
import contextlib, io, json, re, shutil, tempfile, unittest, zipfile
from pathlib import Path
from unittest.mock import patch
import common, experiment, evaluation, reporting

class OfflineChecks(unittest.TestCase):
    def setUp(self):
        self.refs={r['id']:r for r in common.read(common.ROOT/'data/references.json')}

    def test_sources_and_questions(self):
        qs=common.read(common.ROOT/'data/questions.json')
        self.assertEqual(len(qs),5)
        self.assertTrue(all(set(q)=={'id','question'} for q in qs))
        chunks={c['id']:c for c in common.read(common.ROOT/'data/chunks.json')}
        self.assertEqual(len(chunks),44)
        for ref in self.refs.values():
            for g in ref['evidence_groups']:
                self.assertTrue(g['acceptable_chunk_ids'])
                for cid in g['acceptable_chunk_ids']: self.assertIn(cid,chunks)
                for example in g['source_examples']:
                    self.assertTrue(any(evaluation.compact(example['quote']) in evaluation.compact(chunks[cid]['text']) for cid in g['acceptable_chunk_ids']))

    def test_e5_paraphrases(self):
        for text in ['확정 입주일은 자료에 나와 있지 않습니다.','입주일은 명시되어 있지 않습니다.',
                     '확정 입주일은 제공된 자료로 확인할 수 없습니다.','입주일 정보가 없습니다.']:
            self.assertTrue(evaluation.unknown_notice(text,'입주'),text)
        for text in ['입주일은 2030년입니다.','입주일 정보가 없는 것은 아닙니다.','입주일이 없지 않습니다.']:
            self.assertFalse(evaluation.unknown_notice(text,'입주'),text)

    def test_year_criterion(self):
        ref={'required_full_years':[2030]}
        self.assertEqual(evaluation.year_score('2030년 내 착공 목표입니다.',ref)['value'],1)
        self.assertEqual(evaluation.year_score('’30년 내 착공 목표입니다.',ref)['value'],0)
        self.assertEqual(evaluation.year_score('30년 내 착공 목표입니다.',ref)['value'],0)
        self.assertIsNone(evaluation.year_score('20년 거주 후 분양전환',{})['value'])

    def test_entity_swap_does_not_pass(self):
        good='구리토평2는 1.85만호, 구리건원은 1.46만호입니다.'
        bad='구리토평2는 1.46만호, 구리건원은 1.85만호입니다.'
        self.assertEqual(evaluation.element_score(good,self.refs['Q3'])['value'],1)
        self.assertEqual(evaluation.element_score(bad,self.refs['Q3'])['value'],0)

    def test_expected_partial_answer(self):
        text='하남미사 K-컬처는 1.2만호, 하남감일2는 6.5천호입니다. 두 지구 모두 확정 입주일은 자료에 나와 있지 않습니다.'
        self.assertEqual(evaluation.element_score(text,self.refs['Q5'])['value'],1)

    def test_unknown_price_with_invented_price_not_pass(self):
        text='확정 분양가는 확인할 수 없지만 5억원입니다.'
        self.assertEqual(evaluation.element_score(text,self.refs['Q4'])['value'],0)

    def test_decomposition_validation(self):
        question=self.refs['Q3']['question']
        self.assertEqual(experiment.validate_subqueries('{"queries":["2023년 구리토평2 공급 규모는?"]}',question),['2023년 구리토평2 공급 규모는?'])
        with self.assertRaises(ValueError): experiment.validate_subqueries('{"queries":["구리토평2는 18500호인가요?"]}',question)
        with self.assertRaises(ValueError): experiment.validate_subqueries('{"queries":"wrong"}',question)

    def test_context_budget_and_unique_ids(self):
        chunks=common.read(common.ROOT/'data/chunks.json'); by_id={c['id']:c for c in chunks}
        ranked=[{'id':c['id'],'score':1/(i+1)} for i,c in enumerate(chunks)]
        selected,context,_=experiment.select_context(ranked,by_id)
        self.assertLessEqual(len(context),4200); self.assertLessEqual(len(selected),8)
        self.assertEqual(len({c['id'] for c in selected}),len(selected))

    def test_pipeline_resume_provenance_and_export(self):
        with tempfile.TemporaryDirectory(prefix='week04_offline_') as folder:
            root=Path(folder)
            shutil.copytree(common.ROOT/'src',root/'src')
            (root/'data').mkdir()
            for p in (common.ROOT/'data').glob('*.json'): shutil.copyfile(p,root/'data'/p.name)
            shutil.copyfile(common.ROOT/'README.template.md',root/'README.template.md')
            run=root/'results'/'OFFLINE_TEST_ONLY'; run.mkdir(parents=True)
            seen=[]
            class FakeRetriever:
                def __init__(self,r):
                    self.by_id={c['id']:c for c in common.read(root/'data/chunks.json')}
                def load_reranker(self): pass
                def search(self,q,qid,variant,counters):
                    seen.append(('search',qid,variant))
                    counters['dense_queries']+=1
                    counters['bm25_queries']+=1
                    ranked=[{'id':cid,'score':1/(i+1)} for i,cid in enumerate(list(self.by_id)[:10])]
                    return {'query':q,'ranked_candidates':ranked}
                def rerank(self,q,ranked,counters):
                    counters['rerank_pairs']+=len(ranked); counters['rerank_batches']+=1
                    return list(reversed(ranked)),[]
            def fake_api(r,endpoint,payload,phase,purpose,qid,variant):
                seen.append(('answer',qid,variant))
                return {'done':True,'done_reason':'stop','message':{'content':'OFFLINE TEST ONLY'},'prompt_eval_count':10,'eval_count':4}
            patches=[patch.object(module,'ROOT',root) for module in (common,experiment,evaluation,reporting)]
            with contextlib.ExitStack() as stack:
                for p in patches: stack.enter_context(p)
                common.write(run/'manifest.json',{'policy_sha256':common.policy_hash()})
                common.write(root/'results/latest.json',{'relative_path':'results/OFFLINE_TEST_ONLY'})
                stack.enter_context(patch.object(experiment,'Retriever',FakeRetriever))
                stack.enter_context(patch.object(experiment,'api',fake_api))
                stack.enter_context(patch.object(experiment,'decompose',lambda *args: (['공급 규모는?'],{},{})))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                experiment.run_all()
                records=common.read(run/'records.json')
                self.assertEqual(len(records),15)
                self.assertTrue(all(r['status']=='completed' for r in records))
                self.assertEqual(sum(1 for x in seen if x[0]=='answer'),15)
                self.assertEqual(sum(1 for x in seen if x[0]=='search'),20)
                n=len(seen); experiment.run_all(); self.assertEqual(len(seen),n)
                for rec in records:
                    self.assertEqual(evaluation.actual_context(rec),rec['actual_context'])
                    self.assertNotIn('expected_answer',rec['actual_request']['messages'][-1]['content'])
                broken=dict(records[0]); broken['actual_context']='tampered'
                with self.assertRaises(ValueError): evaluation.actual_context(broken)
                archive=reporting.export(run)
                with zipfile.ZipFile(archive) as z:
                    self.assertIn('README.md',z.namelist()); self.assertIn('results/OFFLINE_TEST_ONLY/records.json',z.namelist())
                (root/'data/questions.json').write_text('[]',encoding='utf-8')
                with self.assertRaises(RuntimeError): common.verify(run)

if __name__=='__main__': unittest.main(verbosity=2)
