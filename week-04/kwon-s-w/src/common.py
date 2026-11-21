"""Standard-library experiment I/O. No API keys or reference labels in generation."""
from pathlib import Path
import hashlib, json, os, time, urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('RAGAS_DO_NOT_TRACK', 'true')
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
CONFIG = {
    'generation_model': 'gemma3:4b', 'embedding_model': 'embeddinggemma',
    'temperature': 0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 600,
    'dense_candidates': 6, 'lexical_candidates': 6, 'rrf_k': 60,
    'final_top_k': 8, 'context_max_characters': 4200,
    'reranker': 'BAAI/bge-reranker-v2-m3', 'reranker_device': 'cpu',
    'reranker_max_length': 1024, 'reranker_batch_size': 2,
    'max_subqueries': 3, 'decomposition_num_predict': 384,
    'judge_num_predict': 2048, 'timeout_seconds': 600,
    'repetitions': 1,
}
VARIANTS = ['baseline', 'rerank', 'decompose']
LABELS = {'baseline': '기준선', 'rerank': '재순위화', 'decompose': '질문 분해'}
BASE = 'http://127.0.0.1:11434'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A local atomic replace is not reliable on every mounted Drive filesystem.
    # Keep a previous valid snapshot and write a complete JSON document.
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    if path.exists():
        try:
            read(path)
            path.with_suffix(path.suffix + '.bak').write_bytes(path.read_bytes())
        except (ValueError, OSError):
            pass
    path.write_text(text, encoding='utf-8')

def sha(value):
    if not isinstance(value, bytes):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()

def digest(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True))

def latest():
    marker = ROOT / 'results/latest.json'
    if not marker.exists():
        raise RuntimeError('먼저 4번 준비 셀을 실행하세요.')
    return ROOT / read(marker)['relative_path']

def policy_hash():
    files = sorted((ROOT / 'src').glob('*.py'))
    files += sorted((ROOT / 'data').glob('*.json'))
    files += sorted((ROOT / 'data' / 'pdf').glob('*.pdf'))
    return digest({p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in files})

def verify(run):
    expected = read(run / 'manifest.json')['policy_sha256']
    if policy_hash() != expected:
        raise RuntimeError('코드/자료/평가 기준이 실행 시작 후 바뀌었습니다. 기존 실행에 섞지 마세요.')

def api(run, endpoint, payload, phase, purpose, question_id=None, variant=None):
    import requests
    item = {'phase': phase, 'purpose': purpose, 'question_id': question_id,
            'variant': variant, 'endpoint': endpoint, 'request': payload,
            'started_at_unix': time.time(), 'response': None, 'error': None}
    started = time.perf_counter()
    try:
        response = requests.post(BASE + '/api/' + endpoint, json=payload,
                                 timeout=CONFIG['timeout_seconds'])
        response.raise_for_status()
        item['response'] = response.json()
        return item['response']
    except Exception as exc:
        item['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        item['wall_seconds'] = time.perf_counter() - started
        with (run / 'calls.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')

def ensure_finished(data, reserve):
    if not data.get('done') or data.get('done_reason') != 'stop':
        raise ValueError('모델 응답 미완료/출력 제한. 원 응답은 calls.jsonl에 보존했습니다.')
    if (data.get('prompt_eval_count') or 0) + reserve > CONFIG['num_ctx']:
        raise ValueError('입력+출력 예약량이 컨텍스트 예산을 초과했습니다.')
    answer = data.get('message', {}).get('content', '').strip()
    if not answer:
        raise ValueError('빈 응답')
    return answer

def metric(value=None, reason='', status='measured', **details):
    return {'value': value, 'status': status, 'reason': reason, **details}

def display_metric(item):
    if item is None:
        return '미실행'
    if item['value'] is None:
        return '해당 없음' if item['status'] == 'not_applicable' else '오류/미측정'
    return f"{item['value']:.2f}"
