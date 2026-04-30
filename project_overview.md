# 프로젝트 개요

---

## 📁 데이터셋

### STT 학습용 데이터

| 항목 | 링크 |
|------|------|
| AI Hub 통화 데이터 | [바로가기](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=통화&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=571) |
| 금융감독원 데이터 | [바로가기](https://www.fss.or.kr/fss/bbs/B0000203/list.do?menuNo=200686&bbsId=&cl1Cd=&pageIndex=1&sdate=&edate=&searchCnd=1&searchWrd=) |

### 딥보이스 실험용 데이터

| 항목 | 링크 |
|------|------|
| 감성 및 발화 스타일별 음성합성 데이터 (보조) | [바로가기](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=감성+및+발화+스타일별+음성합성+데이터&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=466) |
| ASVspoof | [바로가기](https://www.asvspoof.org/) |

---

## 🎙️ Streaming STT 파이프라인 (담당: 김주원)

### 파이프라인 흐름

```
[1. Mic Input]
    └─ 스피커폰 ON → 마이크 입력 → 실시간 오디오 스트림 생성
           ↓
[2. Audio Stream Capturer]
    └─ 오디오를 끊김 없이 STT로 전달 (streaming 방식)
           ↓
[3. Streaming STT Engine]
    ├─ 입력: audio stream
    ├─ 출력: partial (실시간 중간 결과) + final (확정 문장)
    └─ 예시)
        PARTIAL: "안녕하세요 고객님"
        PARTIAL: "안녕하세요 고객님 지금"
        FINAL:   "안녕하세요 고객님 지금 확인해드리겠습니다"
           ↓
[4. Text Stream Manager]
    ├─ live buffer: 실시간 말하는 중 (partial)
    └─ history: 확정 문장 (final 저장)
           ↓
[5. Output / 전달 모듈]
    └─ 다른 모듈로 텍스트 전달
```

---

## 🤖 NLP 분석 파이프라인 (담당: 박가은)

| 단계 | 모델 / 도구 | 역할 |
|------|------------|------|
| 1단계 | **KoELECTRA** | 문맥 판단 (한국어 특화 경량화 모델) |
| 2단계 | **LLM API** | 정밀 분석 · 의심사만 처리 |
| 3단계 | **라벨 수집** | KoBERT 학습 데이터 구축 |
| 4단계 | **KoBERT TFLite** | 최종 배포 모델 (앱 탑재용 용량 압축 모델) |

학습 데이터는 MP4 
→
 ffmpeg → Whisper → KoNLPy 순으로 파싱하고, 
 실제 앱에서는 Android SpeechRecognizer로 실시간 텍스트를 받아 KoNLPy 
 → 
 KoELECTRA tokenizer로 처리합니다.




---

## 🔊 딥보이스 탐지 (담당: 반소람, 김용준)

### 개발 단계

#### 1. 데이터셋 확보 및 정제
- 진짜 목소리 데이터 / 가짜 목소리 데이터 구분하여 확보
- 무음 구간 제거 또는 라벨링
  - 진짜: `0`
  - 가짜: `1`

#### 2. RawNet2 구축
- RawNet2 모델 (오픈소스) 활용
- 음성 데이터 입력 시 확률값 출력되도록 연결

#### 3. 실행 환경 고정
- Docker 등을 활용하여 실행 환경 고정

#### 4. EER 낮추기
- 모델 구축 및 실행 환경 구성 완료 후
- 오류율(EER, Equal Error Rate) 최소화




## [AI 분석 결과 패널]
    ├─ 위험도 점수
    ├─ 의심 키워드
    └─ 경고 상태

