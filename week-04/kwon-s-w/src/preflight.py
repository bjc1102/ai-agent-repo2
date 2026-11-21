"""Real small requests on Colab; separate from the five scored questions."""
import time, subprocess
from common import CONFIG, latest, verify, api, write, ensure_finished

def main():
    run=latest(); verify(run)
    started=time.perf_counter()
    print('GPU 사전 점검: 실제 임베딩 1회와 짧은 답변 1회. 비교 점수에는 포함하지 않습니다.',flush=True)
    vector=api(run,'embed',{'model':CONFIG['embedding_model'],'input':['김포아라 공급 규모는 3.7만호이다.'],'truncate':False},'preflight','embedding')
    if not vector.get('embeddings') or not vector['embeddings'][0]:
        raise RuntimeError('임베딩 사전 점검 실패')
    print('임베딩 응답 확인. Gemma GPU 답변을 기다리는 중입니다.',flush=True)
    data=api(run,'chat',{'model':CONFIG['generation_model'],'stream':False,
        'options':{'num_ctx':CONFIG['num_ctx'],'num_predict':128,'temperature':0,'seed':42},
        'messages':[{'role':'system','content':'참고자료만 근거로 한국어 한 문장으로 답하세요.'},
                    {'role':'user','content':'참고자료: 김포아라 공급 규모는 3.7만호이다. 질문: 김포아라 공급 규모는?'}]},
        'preflight','short_answer')
    answer=ensure_finished(data,128)
    result={'status':'completed','answer':answer,'wall_seconds':time.perf_counter()-started,
            'note':'Endpoint execution check, not a quality score or one of the five benchmark questions.'}
    write(run/'preflight.json',result)
    ps=subprocess.run(['ollama','ps'],capture_output=True,text=True).stdout
    write(run/'gpu_preflight.json',{'ollama_ps':ps,'answer':answer})
    print(ps,flush=True)
    if 'GPU' not in ps:
        raise RuntimeError('GPU가 할당됐지만 Ollama의 GPU 사용을 확인하지 못했습니다. GPU 런타임/서버 로그를 확인하세요.')
    print('Gemma 응답:',answer,flush=True)
    print('사전 점검 완료. 실제 답변 정확성을 보증하는 검사는 아닙니다. 다음 비교 셀을 실행하세요.',flush=True)

if __name__=='__main__': main()
