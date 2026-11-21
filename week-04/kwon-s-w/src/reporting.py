"""Offline report/export: runs without GPU, Ollama, Ragas, or torch."""
from __future__ import annotations
import argparse, collections, csv, io, json, statistics, zipfile
from common import ROOT, CONFIG, VARIANTS, LABELS, read, write, digest, latest, verify, display_metric

def optional(path,default):
    return read(path) if path.exists() else default

def calls(run):
    items=[]; invalid=0
    if (run/'calls.jsonl').exists():
        for line in (run/'calls.jsonl').read_text(encoding='utf-8').splitlines():
            try: items.append(json.loads(line))
            except ValueError: invalid+=1
    return items,invalid

def accounting(run):
    items,invalid=calls(run); totals={}
    for item in items:
        key=(item['phase'],item.get('variant') or 'shared',item.get('question_id') or 'shared')
        row=totals.setdefault(key,{'phase':key[0],'variant':key[1],'question_id':key[2],
              'http_calls':0,'chat_calls':0,'embedding_calls':0,'input_tokens':0,'output_tokens':0,
              'embedding_prompt_tokens':0,'wall_seconds':0.,'errors':0})
        response=item.get('response') or {}
        row['http_calls']+=1; row['wall_seconds']+=item['wall_seconds']
        row['errors']+=bool(item.get('error'))
        if item['endpoint']=='chat':
            row['chat_calls']+=1
            row['input_tokens']+=response.get('prompt_eval_count',0) or 0
            row['output_tokens']+=response.get('eval_count',0) or 0
        elif item['endpoint']=='embed':
            row['embedding_calls']+=1
            row['embedding_prompt_tokens']+=response.get('prompt_eval_count',0) or 0
    result={'groups':list(totals.values()),'invalid_log_lines':invalid,
        'external_api_fee_usd':0,'compute_cost_money':None,
        'notes':['Ollama local inference; paid API is not used. Colab/hardware resource costs are not priced.',
                 'setup, rag, evaluation separated; all logged attempts including errors/retries counted.',
                 'wall_seconds here is HTTP time, not the full RAG pipeline time; see records.json.',
                 'Embedding token counts are only those actually reported by Ollama; no estimates added.']}
    write(run/'usage.json',result); return result

def valid_human(run,records):
    human=optional(run/'human_review.json',{'reviews':[],'decisions':{},'analysis':''})
    by_key={r['key']:r for r in records}
    valid={}
    for item in human.get('reviews',[]):
        rec=by_key.get(item['key'])
        if rec and item.get('answer_sha256')==rec['answer_sha256'] and item.get('context_sha256')==rec.get('context_sha256'):
            valid[item['key']]=item
    return human,valid

def build_report(run=None):
    run=run or latest(); verify(run)
    records=optional(run/'records.json',[])
    evaluations={e['key']:e for e in optional(run/'evaluations.json',[])}
    refs={r['id']:r for r in read(ROOT/'data/references.json')}
    human,reviews=valid_human(run,records)
    usage=accounting(run)
    text=['## 실제 비교 결과','',f'실행 ID: `{run.name}`. RAG 기록 {len(records)}/15, 유효한 사람 판정 {len(reviews)}/15.',
          '', '한 번씩 실행한 작은 평가셋입니다. 필수요소 포함률·Faithfulness·근거 Recall을 전체 정답률로 해석하지 않습니다.',
          '', '| 문항 | 구성 | 실행 | 전달 근거 Recall | 필수요소 | 연도 명확성 | Faithfulness | RAG 초 | 사람 판정 |',
          '|---|---|---|---:|---:|---:|---:|---:|---|']
    by_key={r['key']:r for r in records}
    verdicts={'pass':'충족','partial':'부분 충족','fail':'실패','error':'실행 오류'}
    metrics_names=['recall_delivered','elements','year_clarity','faithfulness']
    for qid in refs:
        for variant in VARIANTS:
            key=qid+':'+variant; rec=by_key.get(key)
            metrics=evaluations.get(key,{}).get('metrics',{})
            values=[display_metric(metrics.get(name)) for name in metrics_names]
            verdict=verdicts.get(reviews.get(key,{}).get('verdict'),'미판정')
            seconds=f"{rec['wall_seconds']:.2f}" if rec else '미실행'
            text.append(f"| {qid} | {LABELS[variant]} | {rec['status'] if rec else '미실행'} | "+' | '.join(values)+f' | {seconds} | {verdict} |')
    summary={'run_id':run.name,'variant_summaries':{},'human_reviewed_n':len(reviews),'expected_records':15}
    text+=['','### 구성별 집계','','평균은 측정된 값만 포함하며 오류·적용 불가·미실행을 임의로 0점 처리하지 않았습니다.',
           '', '| 구성 | 지표 | 평균 | 집계 문항 수 / 전체 |','|---|---|---:|---:|']
    for variant in VARIANTS:
        vr=[r for r in records if r['variant']==variant]
        sm={'recorded_n':len(vr),'completed_n':sum(r['status']=='completed' for r in vr),'metrics':{}}
        for name in metrics_names:
            values=[e['metrics'][name]['value'] for e in evaluations.values() if e['variant']==variant and e['metrics'][name]['value'] is not None]
            avg=statistics.mean(values) if values else None
            sm['metrics'][name]={'mean':avg,'measured_n':len(values),'total_n':5}
            text.append(f"| {LABELS[variant]} | {name} | {avg:.2f} | {len(values)}/5 |" if avg is not None else f"| {LABELS[variant]} | {name} | 미측정 | 0/5 |")
        hs=[h for key,h in reviews.items() if key.endswith(':'+variant)]
        sm['human_reviewed_n']=len(hs)
        sm['human_full_success_rate']=sum(h['verdict']=='pass' for h in hs)/5 if len(hs)==5 else None
        sm['mean_rag_seconds']=statistics.mean(r['wall_seconds'] for r in vr if r['status']=='completed') if any(r['status']=='completed' for r in vr) else None
        sm['latency_measured_n']=sum(r['status']=='completed' for r in vr)
        summary['variant_summaries'][variant]=sm
        if len(hs)==5: text.append(f"| {LABELS[variant]} | 사람 기준 완전 충족률 | {sm['human_full_success_rate']:.2f} | 5/5 |")
        else: text.append(f"| {LABELS[variant]} | 사람 기준 완전 충족률 | 판정 미완료 | {len(hs)}/5 |")
    text+=['','### 실행 부담','','검색 1회는 벡터 검색 1회와 BM25 검색 1회로 구성됩니다. 모델 로드·인덱스 준비는 공통/구성별 준비 기록에 별도 저장합니다.',
           '', '| 구성 | RAG chat 호출 | 질문 임베딩 호출 | 검색 질의 수 | 재순위화 쌍 수 | 평가 chat 호출 |',
           '|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        rag=[g for g in usage['groups'] if g['phase']=='rag' and g['variant']==variant]
        judge=[g for g in usage['groups'] if g['phase']=='evaluation' and g['variant']==variant]
        vr=[r for r in records if r['variant']==variant]
        text.append(f"| {LABELS[variant]} | {sum(g['chat_calls'] for g in rag)} | {sum(g['embedding_calls'] for g in rag)} | {sum(r['counts']['retrieval_queries'] for r in vr)} | {sum(r['counts']['rerank_pairs'] for r in vr)} | {sum(g['chat_calls'] for g in judge)} |")
    text+=['','호출·토큰은 `usage.json`에 단계·구성·문항별로 저장됩니다. 호출 집계에는 오류/재시도가 포함되며 검색 질의 수와 RAG 초는 최신 시도의 기록입니다. 이전 시도는 별도 보존합니다.',
           '외부 유료 API 사용료: 0 USD. Colab/전력/하드웨어 비용은 금액 미측정이며 자원 사용이 무료라는 의미가 아닙니다.','',
           '### 문항별 실제 답변과 판정','']
    for qid,ref in refs.items():
        text += [f'#### {qid}', '', '**질문:** '+ref['question'], '', '**기대 답변:** '+ref['expected_answer'],
                 '', '**근거:** '+ref['source_notes'], '']
        for variant in VARIANTS:
            key=qid+':'+variant; rec=by_key.get(key)
            text += ['**'+LABELS[variant]+'**','']
            if not rec: text+=['미실행','']; continue
            text += ['```text',rec['answer'] or rec.get('error') or '(빈 답변)','```','']
            text += ['선택 근거: '+', '.join(f"{c['id']} (PDF {c['page']}쪽)" for c in rec.get('selected_chunks',[])), '']
            for name,item in evaluations.get(key,{}).get('metrics',{}).items():
                text.append(f"- {name}: {display_metric(item)} — {item['reason']}")
            hr=reviews.get(key)
            text += ['','사람 판정: '+(verdicts[hr['verdict']]+' — '+hr['reason'] if hr else '미판정'),'']
            if rec.get('subqueries'): text += ['분해 질문: '+json.dumps(rec['subqueries'],ensure_ascii=False),'']
            if rec.get('warnings'): text += ['주의: '+' / '.join(rec['warnings']),'']
    text += ['### 대표 사례·성공 문항 회귀 확인·우선 개선','',human.get('analysis') or '사람이 원문과 세 구성의 답변을 대조한 분석을 아직 작성하지 않았습니다.','',
             '### 기법 적용 여부 판단','']
    for variant in ('rerank','decompose'):
        decision=human.get('decisions',{}).get(variant,{})
        text += [f"- {LABELS[variant]}: {decision.get('decision') or '미판단'} — {decision.get('reason') or '실제 결과 대조 후 작성'}"]
    write(run/'summary.json',summary)
    report='\n'.join(text)+'\n'
    (run/'report.md').write_text(report,encoding='utf-8')
    template=(ROOT/'README.template.md').read_text(encoding='utf-8')
    (ROOT/'README.md').write_text(template.replace('{{RESULTS}}',report),encoding='utf-8')
    stream=io.StringIO(newline=''); writer=csv.writer(stream)
    writer.writerow(['question_id','variant','status',*metrics_names,'rag_seconds','human_verdict'])
    for rec in records:
        metrics=evaluations.get(rec['key'],{}).get('metrics',{})
        writer.writerow([rec['question_id'],rec['variant'],rec['status'],
            *[metrics.get(n,{}).get('value') for n in metrics_names],rec['wall_seconds'],reviews.get(rec['key'],{}).get('verdict','')])
    (run/'comparison.csv').write_text('\ufeff'+stream.getvalue(),encoding='utf-8')
    return report

def export(run=None,quiet=False):
    run=run or latest(); verify(run)
    destination=ROOT/('week04_kwon-s-w_'+run.name+'.zip')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in ['README.md','README.template.md','requirements.txt','gemma_rag_week04_gpu_clean.ipynb']:
            p=ROOT/name
            if p.is_file(): archive.write(p,name)
        for folder in ('src','data'):
            for p in sorted((ROOT/folder).rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts:
                    archive.write(p,p.relative_to(ROOT).as_posix())
        for p in sorted(run.rglob('*')):
            if p.is_file() and '_index' not in p.parts:
                archive.write(p,p.relative_to(ROOT).as_posix())
        archive.write(ROOT/'results/latest.json','results/latest.json')
    (ROOT/'last_export.txt').write_text(str(destination),encoding='utf-8')
    if not quiet: print('ZIP:',destination)
    return destination

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['report','export'])
    args=parser.parse_args(); build_report()
    if args.stage=='export': export()
    print('README와 보고서 저장 완료. 모델 호출 없음.')
