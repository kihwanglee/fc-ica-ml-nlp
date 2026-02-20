import platform
from kiwipiepy import Kiwi
from gensim import corpora, models
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 플랫폼별 한글 폰트 설정 (한글 깨짐 방지)
_system = platform.system()
if _system == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif _system == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux 등
    _nanum = [f.name for f in fm.fontManager.ttflist if 'Nanum' in f.name]
    if _nanum:
        plt.rcParams['font.family'] = _nanum[0]
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1. 형태소 분석 (명사 추출)
print("[1/5] 형태소 분석기 초기화 중...")
kiwi = Kiwi()
print("      Kiwi 초기화 완료")

def extract_nouns(text):
    result = kiwi.analyze(text)
    nouns = [token.form for token in result[0][0]
             if token.tag.startswith('NN') and len(token.form) > 1]
    return nouns

# 2. 샘플 문서 (2025년 국내 주요 뉴스 헤드라인 50건)
docs = [
    # AI · 기술
    "챗GPT 구독 결제 급증, 국내 생성형 AI 시장 폭발적 성장",
    "한국형 소버린 AI 개발 프로젝트, 네이버·SK텔레콤 등 5개팀 선정",
    "삼성전자 HBM 반도체 엔비디아 납품 승인, 주가 급등",
    "AI 인재 해외 유출 심각, 국내 연구자 연봉 격차 해소 시급",
    "생성형 AI 이미지 일주일에 7억 장 생성, 저작권 논란 확산",
    "국내 AI 스타트업 투자 유치 역대 최고, 글로벌 VC 관심 집중",
    "반도체 수출 규제 강화에 국내 기업 공급망 재편 돌입",
    "AI 반도체 수급 부족으로 스마트폰·PC 가격 인상 전망",
    "딥러닝 기반 신약 개발 플랫폼 임상 성과, 바이오 투자 급증",
    "개인정보 유출 사고 잇달아, 사이버 보안 투자 확대 촉구",
    # 경제 · 금융
    "코스피 4000선 돌파, 사상 첫 '사천피' 시대 개막",
    "한국은행 기준금리 동결 결정, 물가 안정세 확인 후 인하 검토",
    "수출 경상수지 187억 달러 흑자, 반도체·자동차 호조",
    "장바구니 물가 5년 새 23% 급등, 서민 가계 부담 가중",
    "자영업 폐업 역대 최고, 카페·음식점 경영난 심화",
    "청년 실업률 개선에도 쉬는 청년 50만 명 돌파",
    "박사 취득자 30% 미취업, 고학력 일자리 미스매치 심각",
    "사교육비 총액 29조 원 돌파, 물가 상승률 3배 웃돌아",
    "대미 투자 3500억 달러 합의, 한미 정상회담 관세 협상 타결",
    "원달러 환율 급등락 반복, 기업 환리스크 헤지 비상",
    # 부동산
    "서울 아파트 가격 3주 연속 상승폭 둔화, 강남 하락 전환 조짐",
    "금리 인하 기대감에 수도권 아파트 거래량 소폭 반등",
    "전세 사기 피해자 구제 특별법 국회 통과, 보증 한도 확대",
    "1인 가구 증가로 소형 아파트 청약 경쟁률 급등",
    "부동산 PF 부실 우려 확산, 건설사 줄도산 위기 경고",
    # 환경 · 에너지
    "경남 산청 산불 사망자 183명, 역대 최악 산불 피해 기록",
    "지구 평균 기온 파리협약 목표 1.5도 초과, 기후 위기 경보",
    "해수면 온도 역대 최고치 경신, 한반도 폭염 장기화 전망",
    "정부 재생에너지 확대 로드맵 발표, 2030년 태양광·풍력 40% 목표",
    "탄소중립 이행 속도 미흡, 국제사회 한국에 감축 압박 강화",
    # 사회 · 복지
    "출생아 수 9년 만에 반등, 24만 명 넘어 저출생 반전 기대",
    "노인 빈곤율 40% 육박, 기초연금 인상 논의 본격화",
    "10대 스마트폰 과의존 위험군 42%, 청소년 디지털 건강 우려",
    "청소년 자살 사망자 역대 최고, 학교 정신건강 대책 촉구",
    "청년 고립·은둔 2년 새 2배, 사회적 고립 지원 정책 마련",
    "싱크홀 도심 곳곳 발생, 노후 하수관 정비 예산 긴급 투입",
    "항공사 정비 인력 부족 지적, 저가항공 안전 점검 강화",
    "대학 등록금 사립대 75% 인상, 학생·학부모 반발",
    # 스포츠
    "하얼빈 동계 아시안게임 금메달 16개, 한국 종합 2위 쾌거",
    "손흥민 국가대표 은퇴 시사, 후계자 발굴 논의 시작",
    "프로야구 관중 1000만 명 돌파, 역대 최다 관중 기록 경신",
    "류현진 KBO 복귀 후 첫 완투승, 베테랑 건재 과시",
    "파리올림픽 선수단 금메달 13개, 종합 8위 목표 달성",
    "여자 배구 대표팀 아시아 챔피언십 우승, 세계선수권 기대",
    # 정치
    "제21대 대통령 취임, 새 정부 경제 살리기 1호 공약 발표",
    "여야 예산안 협상 타결, 복지·국방·AI 분야 예산 증액",
    "국회 개헌 논의 재점화, 대통령 4년 연임제 개헌안 제출",
    "경주 APEC 정상회의 경주선언 채택, 한국 외교 위상 제고",
    "헌법재판소 선거구 위헌 결정, 국회의원 선거법 개정 착수",
    "고위공직자 부동산 투기 의혹, 공직자 재산 공개 강화 법안 추진",
]

# 3. 전처리
print(f"\n[2/5] 형태소 분석 중... (문서 {len(docs)}건)")
tokenized = [extract_nouns(doc) for doc in docs]
total_nouns = sum(len(t) for t in tokenized)
print(f"      완료 — 문서당 평균 명사 {total_nouns / len(docs):.1f}개 추출")

# 4. 사전 및 행렬 생성
print(f"\n[3/5] 단어 사전 및 문서-단어 행렬 생성 중...")
dictionary = corpora.Dictionary(tokenized)
corpus = [dictionary.doc2bow(doc) for doc in tokenized]
print(f"      완료 — 고유 단어 {len(dictionary)}개, 문서 {len(corpus)}건")

# 5. LDA 모델 학습
print(f"\n[4/5] LDA 모델 학습 중... (토픽 {6}개, passes=50)")
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=6,   # 토픽 수: AI/기술, 경제, 부동산, 환경, 사회, 스포츠/정치
    passes=50,
    random_state=42
)

# 6. 토픽 출력
print(f"      완료\n")
print(f"[5/5] 결과 출력 및 시각화\n")
print(f"{'='*60}")
print(f"  LDA 토픽 분석 결과 — 문서 {len(docs)}건, 토픽 {lda_model.num_topics}개")
print(f"{'='*60}")
for i, topic in lda_model.print_topics(num_words=7):
    print(f"  토픽 {i+1}: {topic}")
print(f"{'='*60}\n")

# 7. 인터랙티브 시각화
output_file = 'lda_result.html'
print(f"      pyLDAvis 시각화 생성 중...")
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, output_file)
print(f"      저장 완료 → {output_file}")

# 8. 분석 결과를 마크다운 파일로 저장
analysis_file = 'lda_analysis.md'
print(f"\n[보너스] 분석 리포트 생성 중...")

NUM_WORDS = 10  # 토픽당 출력할 키워드 수

# 8-1. 각 토픽의 키워드·가중치 추출
topics_data = []
for i in range(lda_model.num_topics):
    top_words = lda_model.show_topic(i, topn=NUM_WORDS)  # [(word, prob), ...]
    topics_data.append(top_words)

# 8-2. 각 문서의 지배 토픽 계산
doc_topic_assignments = []
for i, bow in enumerate(corpus):
    dist = lda_model.get_document_topics(bow, minimum_probability=0)
    dominant = max(dist, key=lambda x: x[1])   # 확률 최대 토픽
    doc_topic_assignments.append((i, dominant[0], dominant[1]))

# 8-3. 마크다운 작성
lines = []
lines.append("# LDA 토픽 모델링 분석 리포트\n")
lines.append(f"- 분석 문서 수: **{len(docs)}건**")
lines.append(f"- 토픽 수: **{lda_model.num_topics}개**")
lines.append(f"- 학습 반복(passes): **{lda_model.passes}**")
lines.append(f"- 고유 단어 수: **{len(dictionary)}개**\n")

lines.append("---\n")
lines.append("## 1. 토픽별 핵심 키워드\n")
for i, top_words in enumerate(topics_data):
    keyword_str = ", ".join(f"{w}({p:.3f})" for w, p in top_words)
    lines.append(f"### 토픽 {i+1}")
    lines.append(f"| 순위 | 키워드 | 가중치 |")
    lines.append(f"|------|--------|--------|")
    for rank, (word, prob) in enumerate(top_words, 1):
        lines.append(f"| {rank} | {word} | {prob:.4f} |")
    lines.append("")

lines.append("---\n")
lines.append("## 2. 문서별 지배 토픽\n")
lines.append("| 문서 번호 | 뉴스 헤드라인 | 지배 토픽 | 확률 |")
lines.append("|-----------|--------------|-----------|------|")
for doc_idx, topic_idx, prob in doc_topic_assignments:
    headline = docs[doc_idx]
    lines.append(f"| {doc_idx+1} | {headline} | 토픽 {topic_idx+1} | {prob:.3f} |")

lines.append("\n---\n")
lines.append("## 3. 토픽별 문서 목록\n")
for t in range(lda_model.num_topics):
    assigned = [(idx, prob) for idx, tid, prob in doc_topic_assignments if tid == t]
    lines.append(f"### 토픽 {t+1} ({len(assigned)}건)\n")
    for doc_idx, prob in assigned:
        lines.append(f"- ({prob:.3f}) {docs[doc_idx]}")
    lines.append("")

with open(analysis_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"      저장 완료 → {analysis_file}")