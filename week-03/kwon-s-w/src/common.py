
"""Shared I/O and native local Ollama calls. No API keys or patch loading."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import unicodedata
import re

os.environ.setdefault('RAGAS_DO_NOT_TRACK', 'true')
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
PROJECT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:11434'
CONFIG = {
    'generation_model': 'gemma3:4b', 'embedding_model': 'embeddinggemma',
    'temperature': 0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 600,
    'distance': 'cosine', 'dense_candidates': 6, 'lexical_candidates': 6,
    'rrf_k': 60, 'final_top_k': 8, 'context_max_characters': 4200,
    'chunk_max_characters': 900, 'tail_sentence_overlap_max': 120,
    'chunk_policy': 'reuse the 29 week02 submitted redacted chunks',
    'embedding_input_format': 'raw text, same convention as week02',
    'query_decomposition': False,
    'generation_output': 'plain answer; retrieved source locations logged separately',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def norm(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def latest():
    marker = PROJECT / 'results/latest_run.json'
    if not marker.exists():
        raise RuntimeError('먼저 2번 RAG 실행 셀을 실행하세요.')
    return PROJECT / read(marker)['relative_path']


def call(run, endpoint, payload, purpose):
    import requests
    trace = {'purpose': purpose, 'endpoint': endpoint, 'request': payload, 'ok': False}
    started = time.perf_counter()
    try:
        response = requests.post(BASE + '/' + endpoint, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        trace.update(response=data, ok=True)
        return data
    except Exception as exc:
        trace['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        trace['wall_seconds'] = time.perf_counter() - started
        with (Path(run) / 'calls.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + '\n')


def metric(value=None, reason='', status='measured', **extra):
    return dict(value=value, reason=reason, status=status, **extra)


def show(metric_record):
    if metric_record['value'] is None:
        return '해당 없음' if metric_record['status'] == 'not_applicable' else '미측정/오류'
    return f'{metric_record["value"]:.2f}'

