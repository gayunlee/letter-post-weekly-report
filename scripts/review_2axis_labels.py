"""
2축(Topic × Sentiment) 라벨링 데이터 전수검사 스크립트

labeled_2axis.json의 3,641건에 대해 의도 기반 규칙을 적용하여
Topic과 Sentiment를 검증하고 수정한다.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data" / "training_data" / "labeled_2axis.json"
OUTPUT_FILE = BASE_DIR / "data" / "training_data" / "labeled_2axis_reviewed.json"
UNCERTAIN_FILE = BASE_DIR / "data" / "training_data" / "uncertain_items.json"

# ============================================================
# 1. 키워드 / 패턴 사전
# ============================================================

# 마스터 콘텐츠 관련 키워드 (강의, 리포트, 방송 등에 대한 반응)
CONTENT_REACTION_PATTERNS = [
    r'강의.*(?:잘|감사|도움|좋|덕분)',
    r'(?:잘|감사|도움|좋|덕분).*강의',
    r'리포트.*(?:잘|감사|도움|좋|덕분)',
    r'(?:잘|감사|도움|좋|덕분).*리포트',
    r'방송.*(?:잘|감사|도움|좋|덕분)',
    r'(?:잘|감사|도움|좋|덕분).*방송',
    r'분석.*(?:잘|감사|도움|좋|덕분)',
    r'(?:잘|감사|도움|좋|덕분).*분석',
    r'영상.*(?:잘|감사|도움|좋|덕분)',
    r'(?:잘|감사|도움|좋|덕분).*영상',
    r'쿠키.*(?:잘|감사|도움|좋|덕분)',
    r'(?:잘|감사|도움|좋|덕분).*쿠키',
    r'(?:잘|감사|도움|좋|덕분).*(?:올려|써|적어)',
    r'글.*(?:잘|감사|도움|좋|덕분)',
    r'콘텐츠.*(?:잘|감사|도움|좋|덕분)',
    r'교육.*(?:잘|감사|도움|좋|덕분)',
    r'라이브.*(?:잘|감사|도움|좋|덕분)',
]

# 마스터를 지칭하는 패턴 (콘텐츠 반응의 단서)
MASTER_MENTION_PATTERNS = [
    r'(?:쌤|선생님|교수님|작가님|대표님|원장님|마스터님|님).*(?:덕분|감사|감동|좋|배우|공부|성장)',
    r'(?:덕분|감사|감동|좋|배우|공부|성장).*(?:쌤|선생님|교수님|작가님|대표님|원장님|마스터님)',
]

# 마스터 콘텐츠에 대한 감사/감동 (text에 마스터 언급 + 감사)
MASTER_GRATITUDE_PATTERNS = [
    r'(?:쌤|선생님|교수님|작가님|대표님|원장님|마스터님|님).*(?:덕분|감사합니다|고맙습니다|고마워요|감동)',
    r'(?:덕분에|감사합니다|고맙습니다|고마워요|감동).*(?:쌤|선생님|교수님|작가님|대표님|원장님|마스터님)',
    r'(?:만나서|만난것|알게되어|알게돼서).*(?:감사|다행|행복|행운|복)',
    r'(?:삶|인생|투자).*(?:변|바뀌|달라).*(?:감사|덕분)',
    r'(?:배우|공부|성장).*(?:감사|덕분)',
]

# 투자 토론/질문/매매 보고 패턴
INVESTMENT_DISCUSSION_PATTERNS = [
    r'(?:PER|PBR|ROE|ROA|EPS|BPS|시총|시가총액|배당)',
    r'(?:매수|매도|손절|익절|물타기|추매|추가매수|분할매수|분할매도|리밸런싱)',
    r'(?:포트폴리오|포폴|비중|섹터|업종|종목).*(?:질문|궁금|어떻|추천|의견)',
    r'(?:질문|궁금|어떻|추천|의견).*(?:포트폴리오|포폴|비중|섹터|업종|종목)',
    r'(?:상승|하락|급등|급락|조정|반등|돌파|지지|저항).*(?:예상|전망|생각|의견)',
    r'(?:수익률|손실률|수익금|손실금|평단|평균단가)',
    r'(?:언제|얼마|몇주|몇퍼).*(?:매수|매도|사|팔)',
    r'(?:보유|매수|매도).*(?:할까|해야|하는게|좋을까)',
    r'(?:목표가|목표주가|적정가|적정주가)',
]

# 투자 주제 키워드 (단어 수준)
INVESTMENT_KEYWORDS = [
    '매수', '매도', '손절', '익절', '물타기', '추매', '추가매수', '분할매수',
    'PER', 'PBR', 'ROE', 'EPS', '시총', '배당', '실적', '어닝',
    '포트폴리오', '포폴', '비중조절', '리밸런싱',
    '상한가', '하한가', '급등', '급락', '갭상', '갭하',
    '차트', '이평선', '볼린저', '거래량', '수급', '외국인', '기관', '연기금',
    '공매도', '신용잔고', '대차잔고',
]

# 특정 종목명 패턴 (투자 이야기의 단서)
STOCK_NAMES = [
    '두산', '에너빌리티', '두빌', '한화오션', '한화', 'HD현대',
    '삼성전자', '삼전', 'SK하이닉스', '하이닉스', 'LG전자', 'LG화학',
    'LG디스플레이', 'LG디플', '현대차', '기아', '파미셀', '엠로',
    'DL이앤씨', '두산우', '한국전력', '한전', '삼성에스피',
    'OCI', '임핀지', '한온시스템', '코스피', '코스닥', '나스닥',
    'S&P', 'ETF',
]

# 커뮤니티 소통 패턴 (인사, 안부, 축하, 위로)
COMMUNITY_PATTERNS = [
    r'안녕하세요[.!?\s]*$',
    r'(?:건강|몸조리|컨디션).*(?:챙기|조심|유의|잘|하세요)',
    r'(?:명복|삼가|애도|위로|추모)',
    r'(?:새해|설|추석|명절|생일|크리스마스|연말).*(?:복|축하|기원)',
    r'(?:축하|응원|파이팅|화이팅|홧팅|힘내|잘|좋|해요|합시다|하세요)[!♡♥]*\s*$',
    r'^(?:축하|응원|파이팅|화이팅|홧팅|힘내)',
    r'(?:반갑습니다|반가워요|잘부탁|잘 부탁)',
    r'(?:다들|모두|학우|여러분).*(?:홧팅|화이팅|파이팅|힘내|좋은|하루)',
    r'(?:좋은|즐거운|행복한).*(?:하루|주말|한주|저녁|아침)',
    r'(?:감기|날씨|추위|더위).*(?:조심|주의|챙기)',
]

# 서비스 이슈 패턴 (보다 엄격하게: 플랫폼/앱 기능, 결제, 구독 운영에 대한 직접적 이슈)
SERVICE_PATTERNS = [
    r'(?:환불|결제|구독|해지|취소).*(?:요청|부탁|해주|안됨|안되|않됨|어떻게)',
    r'(?:앱|어플|사이트|플랫폼|시스템).*(?:오류|에러|버그|안됨|안되|느림|문제|개선)',
    r'(?:세미나|오프라인).*(?:신청|접수|예약|확인).*(?:안됨|안되|어떻게)',
    r'(?:멤버십|회원).*(?:가입|탈퇴|등급|관리).*(?:어떻게|안됨|안되|방법|불만|부탁)',
    r'(?:프리미엄|유료|무료|구독).*(?:가입|방법|어떻게)',
    r'(?:댓글|기능|알림|푸시).*(?:열어|안됨|안되|없|추가|해주)',
    r'(?:어떻게).*(?:가입|신청|결제|구독)',
    r'(?:탈퇴|환불).*(?:합니다|하겠|할게|해주)',
    r'(?:소통|쌍방향|댓글).*(?:안되|안됨|없|열어|해주)',
    r'(?:운영|관리).*(?:안하|부족|부실|아쉽|불만)',
]

# 감정 키워드
POSITIVE_KEYWORDS = [
    '감사', '고맙', '감동', '좋아', '좋은', '최고', '대단', '훌륭', '멋지', '멋진',
    '행복', '기쁘', '기뻐', '즐거', '뿌듯', '설레', '든든', '따뜻', '사랑',
    '칭찬', '존경', '응원', '격려', '파이팅', '화이팅', '홧팅', '힘내',
    '수익', '흑자', '플러스', '올랐', '날아', '축하', '만족', '편안', '안정',
    '복', '건승', '건강하', '축복',
]

NEGATIVE_KEYWORDS = [
    '불만', '실망', '답답', '걱정', '불안', '짜증', '화가', '분노', '속상',
    '힘들', '힘든', '괴로', '고통', '슬프', '서글', '씁쓸', '허탈', '아쉽',
    '불편', '문제', '오류', '에러', '안됨', '환불', '해지',
    '손실', '마이너스', '-', '빠지', '떨어', '폭락', '급락', '하락',
    '죽겄', '죽겠', '못하겠', '무너', '깡통', '물렸',
]

NEUTRAL_INDICATORS = [
    r'\?[\s]*$',  # 물음표로 끝남
    r'(?:궁금|질문|여쭤|여쭈|문의|알려주|설명|부탁)',
    r'(?:인가요|일까요|할까요|되나요|있나요|맞나요|건가요)',
    r'(?:어떻게|어떤|얼마|몇|언제)',
]


# ============================================================
# 2. 판정 함수들
# ============================================================

def has_pattern(text, patterns):
    """텍스트에 패턴 목록 중 하나라도 매치되면 True"""
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False

def count_keywords(text, keywords):
    """텍스트에서 키워드 등장 횟수"""
    count = 0
    for kw in keywords:
        count += len(re.findall(re.escape(kw), text, re.IGNORECASE))
    return count

def has_stock_mention(text):
    """종목명 언급 여부 (닉네임/사람이름에 포함된 종목명은 제외)"""
    # 닉네임 패턴 제거: "OO님", "OO쌤" 등에서 종목명이 포함된 경우
    # e.g. "두산사랑님", "LG팬님" 등
    cleaned = re.sub(r'\S*님', '', text)
    cleaned = re.sub(r'\S*쌤', '', cleaned)
    cleaned = re.sub(r'\S*반트\S*', '', cleaned)  # 블루반트님 등

    for name in STOCK_NAMES:
        if name.lower() in cleaned.lower():
            return True
    return False

def has_question(text):
    """질문 포함 여부"""
    if '?' in text or '요?' in text:
        return True
    question_patterns = [
        r'(?:궁금|질문|여쭤|여쭈|문의)',
        r'(?:인가요|일까요|할까요|되나요|있나요|맞나요|건가요|하나요)',
        r'(?:어떻게|어떤|얼마나|몇|언제).*(?:요|까)',
        r'(?:알려주|설명해|부탁드)',
    ]
    return has_pattern(text, question_patterns)

def is_primarily_investment(text):
    """투자 토론/질문/매매보고가 주 목적인지"""
    inv_score = 0
    # 투자 토론 패턴 매치
    if has_pattern(text, INVESTMENT_DISCUSSION_PATTERNS):
        inv_score += 3
    # 투자 키워드 수
    inv_kw_count = count_keywords(text, INVESTMENT_KEYWORDS)
    inv_score += min(inv_kw_count, 5)
    # 종목명 언급 (닉네임 제외)
    cleaned_text = re.sub(r'\S*님', '', text)
    cleaned_text = re.sub(r'\S*쌤', '', cleaned_text)
    cleaned_text = re.sub(r'\S*반트\S*', '', cleaned_text)
    stock_count = sum(1 for name in STOCK_NAMES if name.lower() in cleaned_text.lower())
    inv_score += min(stock_count * 2, 6)
    # 투자 질문
    if has_question(text) and (inv_kw_count > 0 or stock_count > 0):
        inv_score += 2
    return inv_score

def is_primarily_content_reaction(text):
    """마스터 콘텐츠 반응이 주 목적인지"""
    cr_score = 0
    if has_pattern(text, CONTENT_REACTION_PATTERNS):
        cr_score += 4
    if has_pattern(text, MASTER_GRATITUDE_PATTERNS):
        cr_score += 3
    if has_pattern(text, MASTER_MENTION_PATTERNS):
        cr_score += 2
    # "덕분에" 단독 사용
    if re.search(r'덕분', text):
        cr_score += 1
    return cr_score

def is_primarily_community(text):
    """커뮤니티 소통이 주 목적인지"""
    comm_score = 0
    if has_pattern(text, COMMUNITY_PATTERNS):
        comm_score += 3
    # 짧은 인사/격려
    if len(text) < 30:
        greeting_patterns = [r'화이팅', r'홧팅', r'파이팅', r'힘내', r'감사', r'응원',
                           r'안녕', r'반갑', r'축하', r'새해', r'명복']
        if has_pattern(text, greeting_patterns):
            comm_score += 3
    # 명복/위로
    if re.search(r'(?:명복|삼가|애도|위로|추모|영면)', text):
        comm_score += 5
    # 건강 챙기세요 류
    if re.search(r'(?:건강|몸조리|컨디션).*(?:챙기|하세요|조심)', text):
        comm_score += 3
    # 안부/인사가 주 내용이고 투자 내용이 적은 경우
    return comm_score

def is_primarily_service(text):
    """서비스 이슈가 주 목적인지"""
    svc_score = 0
    if has_pattern(text, SERVICE_PATTERNS):
        svc_score += 4
    # 서비스 이슈 키워드: 단순 "가입", "어플", "멤버십" 등은 맥락에 따라 투자/커뮤니티에서도 사용되므로
    # 서비스 운영 맥락에서만 점수 부여
    strong_svc_keywords = ['환불', '해지', '취소', '오류', '버그', '프리미엄반']
    svc_score += min(count_keywords(text, strong_svc_keywords) * 3, 6)

    # "가입", "멤버십" 등은 서비스 관련 동사와 함께 사용될 때만 점수 부여
    contextual_svc = [
        r'(?:가입|멤버십|구독).*(?:어떻게|방법|안됨|안되|해지|취소|환불)',
        r'(?:어떻게|방법).*(?:가입|멤버십|구독)',
    ]
    if has_pattern(text, contextual_svc):
        svc_score += 2
    return svc_score


def determine_sentiment(text, current_sentiment):
    """Sentiment 재판정"""
    pos_count = count_keywords(text, POSITIVE_KEYWORDS)
    neg_count = count_keywords(text, NEGATIVE_KEYWORDS)
    is_question = has_question(text)

    # 규칙 기반 보정 사유를 함께 반환
    reasons = []

    # 1) 질문이 주 내용이면서 감정 표현이 약한 경우 → 중립
    if is_question and pos_count <= 1 and neg_count == 0:
        if current_sentiment == '긍정':
            reasons.append("질문이 주 내용, 긍정 표현 미약")
            return '중립', reasons

    # 2) 질문이 포함되어도 전체 톤이 감사/긍정이 강하면 → 긍정 유지
    if is_question and pos_count >= 3 and neg_count == 0:
        if current_sentiment == '중립':
            reasons.append("질문 포함이나 긍정 톤이 강함(감사+질문)")
            return '긍정', reasons

    # 3) 부정 키워드가 있지만 격려/응원 맥락이면 → 긍정
    encouragement_patterns = [
        r'(?:힘들|어렵|걱정).*(?:지만|해도|겠지만).*(?:화이팅|홧팅|파이팅|힘내|응원|함께|같이|이겨)',
        r'(?:손실|하락|마이너스).*(?:두려워|걱정).*(?:마세요|말고|않아도)',
        r'(?:힘들|괴로|어렵).*(?:지만|해도).*(?:감사|다행|행복|좋)',
    ]
    if has_pattern(text, encouragement_patterns) and pos_count >= neg_count:
        if current_sentiment == '부정':
            reasons.append("격려/응원 맥락에서 부정 키워드 사용")
            return '긍정', reasons

    # 4) 정중한 불만/요청 (키워드로 안 잡히는 부정)
    polite_complaint_patterns = [
        r'(?:해주셨으면|해주시면|바랍니다|부탁드립니다).*(?:아쉽|개선|불편)',
        r'(?:아쉽|개선|불편).*(?:해주셨으면|해주시면|바랍니다|부탁드립니다)',
        r'(?:왜|어째서).*(?:안|없|못).*(?:주시|해주|하나요)',
        r'(?:답답|불편|아쉽|서운).*(?:의견|말씀|드립니다|봅니다|합니다)',
    ]
    if has_pattern(text, polite_complaint_patterns) and current_sentiment != '부정':
        if neg_count >= 1 or has_pattern(text, polite_complaint_patterns):
            reasons.append("정중한 불만/요청")
            return '부정', reasons

    # 5) 명확한 감정 불일치 보정
    # 긍정으로 되어있지만 실제로는 부정이 훨씬 강한 경우
    if current_sentiment == '긍정' and neg_count >= 3 and pos_count <= 1:
        reasons.append(f"부정 키워드({neg_count})가 긍정({pos_count})보다 훨씬 많음")
        return '부정', reasons

    # 부정으로 되어있지만 실제로는 긍정이 훨씬 강한 경우
    if current_sentiment == '부정' and pos_count >= 3 and neg_count <= 1:
        reasons.append(f"긍정 키워드({pos_count})가 부정({neg_count})보다 훨씬 많음")
        return '긍정', reasons

    # 6) 중립인데 강한 감정이 있는 경우
    if current_sentiment == '중립':
        if pos_count >= 3 and neg_count == 0:
            reasons.append(f"긍정 키워드 다수({pos_count}), 부정 없음")
            return '긍정', reasons
        if neg_count >= 3 and pos_count == 0:
            reasons.append(f"부정 키워드 다수({neg_count}), 긍정 없음")
            return '부정', reasons

    # 7) 긍정인데 감정 표현이 전혀 없고 사실 기술/정보만 있는 경우
    if current_sentiment == '긍정' and pos_count == 0 and neg_count == 0:
        # 이모티콘만 있는 경우는 유지
        if not re.search(r'[♡♥❤💚🍀😁😊☺️👍💪🎉]', text):
            if len(text) > 20:  # 충분히 긴 텍스트인데 감정 표현이 없으면
                reasons.append("감정 표현 없이 사실/정보 기술")
                return '중립', reasons

    # 8) 명확한 손실/고통 표현
    loss_patterns = [
        r'(?:손실|마이너스|-\d+%).*(?:크|커|너무|많|심각|감당)',
        r'(?:깡통|물렸|물려|죽겠|죽겄|못하겠)',
        r'(?:너무|많이|크게).*(?:힘들|괴로|고통|빠지|떨어)',
    ]
    if has_pattern(text, loss_patterns) and current_sentiment != '부정':
        reasons.append("명확한 손실/고통 표현")
        return '부정', reasons

    # 9) 응원/격려 (긍정으로 봐야 함)
    if current_sentiment != '긍정':
        cheer_patterns = [
            r'(?:화이팅|홧팅|파이팅|힘내세요|힘냅시다|응원합니다)',
            r'(?:함께|같이).*(?:이겨|극복|해내|해봐)',
        ]
        if has_pattern(text, cheer_patterns) and neg_count <= 1:
            reasons.append("응원/격려 표현")
            return '긍정', reasons

    return current_sentiment, []


# ============================================================
# 3. 메인 검사 로직
# ============================================================

def review_item(item):
    """
    단일 항목을 검사하여 수정된 topic, sentiment와 수정 사유를 반환.
    Returns: (new_topic, topic_reason, new_sentiment, sentiment_reason, is_uncertain)
    """
    text = item['text']
    orig_cat = item['original_category']
    current_topic = item['topic']
    current_sentiment = item['sentiment']

    # 점수 계산
    inv_score = is_primarily_investment(text)
    cr_score = is_primarily_content_reaction(text)
    comm_score = is_primarily_community(text)
    svc_score = is_primarily_service(text)

    new_topic = current_topic
    topic_reason = None
    is_uncertain = False

    # === Topic 수정 규칙 ===

    # 규칙 1: 커뮤니티 소통인데 투자 토론/질문/매매보고가 주 의도
    if current_topic == '커뮤니티 소통':
        if inv_score >= 5 and inv_score > comm_score + 2:
            new_topic = '투자 이야기'
            topic_reason = f"커뮤니티→투자: 투자점수({inv_score})>커뮤니티점수({comm_score}), 투자토론/질문이 주 의도"
        elif inv_score >= 3 and has_question(text) and has_stock_mention(text):
            new_topic = '투자 이야기'
            topic_reason = f"커뮤니티→투자: 종목 언급 + 질문이 주 내용"
        # 커뮤니티 소통인데 마스터 콘텐츠 반응이 주 의도
        elif cr_score >= 5 and cr_score > comm_score + 2:
            new_topic = '콘텐츠 반응'
            topic_reason = f"커뮤니티→콘텐츠: 콘텐츠반응점수({cr_score})>커뮤니티점수({comm_score})"
        # 커뮤니티 소통인데 서비스 이슈 (보다 엄격한 기준)
        elif svc_score >= 5 and svc_score > comm_score + 2:
            new_topic = '서비스 이슈'
            topic_reason = f"커뮤니티→서비스: 서비스점수({svc_score})>커뮤니티점수({comm_score})"

    # 규칙 2: 콘텐츠 반응인데 투자 토론이 주 내용
    elif current_topic == '콘텐츠 반응':
        if inv_score >= 6 and cr_score <= 2:
            new_topic = '투자 이야기'
            topic_reason = f"콘텐츠→투자: 투자점수({inv_score})높고 콘텐츠반응점수({cr_score})낮음"
        # 콘텐츠 반응인데 커뮤니티 인사가 주 의도
        elif comm_score >= 5 and cr_score <= 2:
            new_topic = '커뮤니티 소통'
            topic_reason = f"콘텐츠→커뮤니티: 커뮤니티점수({comm_score})높고 콘텐츠반응점수({cr_score})낮음"
        # 명복, 건강 챙기세요 등
        elif re.search(r'(?:명복|삼가|애도|위로|추모|영면)', text):
            new_topic = '커뮤니티 소통'
            topic_reason = "콘텐츠→커뮤니티: 조의/위로 표현이 주 목적"
        elif re.search(r'(?:건강|몸조리).*(?:챙기|하세요)', text) and cr_score <= 1:
            new_topic = '커뮤니티 소통'
            topic_reason = "콘텐츠→커뮤니티: 안부 인사가 주 목적"
        # 콘텐츠 반응인데 서비스 이슈 (보다 엄격한 기준)
        elif svc_score >= 5 and cr_score <= 2:
            new_topic = '서비스 이슈'
            topic_reason = f"콘텐츠→서비스: 서비스점수({svc_score})높고 콘텐츠반응점수({cr_score})낮음"
        # 콘텐츠 반응인데 학우/동료에게 감사 (마스터가 아닌 다른 회원에게)
        elif re.search(r'(?:학우|회원|분들?).*(?:감사|고맙|덕분)', text) and not has_pattern(text, MASTER_GRATITUDE_PATTERNS):
            if comm_score >= 2:
                new_topic = '커뮤니티 소통'
                topic_reason = "콘텐츠→커뮤니티: 학우/회원에 대한 감사 (마스터 콘텐츠 반응 아님)"

    # 규칙 3: 투자 이야기인데 실제로는 다른 목적
    elif current_topic == '투자 이야기':
        if inv_score <= 2 and comm_score >= 4:
            new_topic = '커뮤니티 소통'
            topic_reason = f"투자→커뮤니티: 투자점수({inv_score})낮고 커뮤니티점수({comm_score})높음"
        elif inv_score <= 2 and cr_score >= 4:
            new_topic = '콘텐츠 반응'
            topic_reason = f"투자→콘텐츠: 투자점수({inv_score})낮고 콘텐츠반응점수({cr_score})높음"
        elif inv_score <= 2 and svc_score >= 5:
            new_topic = '서비스 이슈'
            topic_reason = f"투자→서비스: 투자점수({inv_score})낮고 서비스점수({svc_score})높음"
        # 커뮤니티 분위기 관련 글 (투자보다 커뮤니티 소통이 주)
        elif re.search(r'(?:분위기|커뮤니티|게시판).*(?:살벌|안좋|힘들|씁쓸)', text) and inv_score <= 3:
            if comm_score >= 2:
                new_topic = '커뮤니티 소통'
                topic_reason = "투자→커뮤니티: 커뮤니티 분위기에 대한 소통이 주 목적"
        # 멤버십/서비스 관련 불만이 주 내용 (투자 점수도 높지 않아야)
        elif re.search(r'(?:멤버십|회원관리|운영|댓글기능|소통).*(?:불만|안됨|안되|없|불편|요청|해주)', text):
            if svc_score >= 3 and inv_score <= 3:
                new_topic = '서비스 이슈'
                topic_reason = "투자→서비스: 서비스/운영 관련 불만이 주 목적"

    # 규칙 4: 서비스 이슈인데 다른 목적
    elif current_topic == '서비스 이슈':
        if svc_score <= 1 and inv_score >= 5:
            new_topic = '투자 이야기'
            topic_reason = f"서비스→투자: 서비스점수({svc_score})낮고 투자점수({inv_score})높음"
        elif svc_score <= 1 and cr_score >= 4:
            new_topic = '콘텐츠 반응'
            topic_reason = f"서비스→콘텐츠: 서비스점수({svc_score})낮고 콘텐츠반응점수({cr_score})높음"

    # === 추가 정밀 규칙 ===

    # 짧은 텍스트(< 20자)에 대한 특별 규칙
    if len(text.strip()) < 20:
        stripped = text.strip()
        # 순수 인사/응원
        if re.search(r'^(?:화이팅|홧팅|파이팅|힘내세요|응원합니다|감사합니다|고맙습니다|반갑습니다|안녕하세요)[!.♡♥]*\s*$', stripped):
            if current_topic not in ('커뮤니티 소통',):
                new_topic = '커뮤니티 소통'
                topic_reason = f"{current_topic}→커뮤니티: 짧은 인사/격려 문구"
        # 이모티콘만
        if re.match(r'^[♡♥❤💚🍀😁😊☺️👍💪🎉\s]+$', stripped):
            if current_topic != '커뮤니티 소통':
                new_topic = '커뮤니티 소통'
                topic_reason = f"{current_topic}→커뮤니티: 이모티콘만 있는 텍스트"

    # "감사합니다 OO님" (특정 회원에게 짧은 감사) → 커뮤니티
    if re.match(r'^(?:감사합니다|감사드립니다|고맙습니다)\s+\S+님\s*$', text.strip()):
        if current_topic == '콘텐츠 반응':
            # 마스터가 아닌 회원에게 감사일 수 있음
            new_topic = '커뮤니티 소통'
            topic_reason = "콘텐츠→커뮤니티: 회원에 대한 짧은 감사"

    # 투자 키워드가 있어도 "마스터님 덕분에 투자가 안정적" 류는 콘텐츠 반응
    if new_topic == '투자 이야기' or current_topic == '투자 이야기':
        gratitude_investment_patterns = [
            r'(?:쌤|선생님|교수님|작가님|대표님|원장님|마스터님).*(?:덕분|감사).*(?:투자|수익|안정|편안|성장)',
            r'(?:덕분|감사).*(?:투자|수익|안정|편안|성장).*(?:했|되었|됐|하고)',
        ]
        if has_pattern(text, gratitude_investment_patterns) and cr_score >= 3:
            if new_topic != '콘텐츠 반응':
                new_topic = '콘텐츠 반응'
                topic_reason = "투자→콘텐츠: 마스터 덕분의 투자 성과 감사가 주 목적"

    # === Sentiment 수정 ===
    new_sentiment, sent_reasons = determine_sentiment(text, current_sentiment)
    sentiment_reason = sent_reasons[0] if sent_reasons else None

    # 특수 케이스: 서비스 불만인데 긍정/중립으로 되어있는 경우
    if new_topic == '서비스 이슈' and orig_cat in ('서비스 불편사항', '서비스 피드백'):
        if new_sentiment == '긍정':
            complaint_check = re.search(r'(?:불편|불만|안됨|안되|환불|해지|오류|문제)', text)
            if complaint_check:
                new_sentiment = '부정'
                sentiment_reason = "서비스 불만 내용인데 긍정으로 분류됨"

    # 커뮤니티 소통에서 위로/격려의 맥락
    if new_topic == '커뮤니티 소통':
        # 위로 맥락에서 "힘들" 등이 있어도 전체 톤이 격려면 긍정
        if new_sentiment == '부정' and has_pattern(text, [
            r'(?:힘들|어렵|걱정).*(?:화이팅|홧팅|파이팅|힘내|함께|응원)',
            r'(?:화이팅|홧팅|파이팅|힘내|함께|응원).*(?:힘들|어렵|걱정)',
        ]):
            pos_c = count_keywords(text, POSITIVE_KEYWORDS)
            neg_c = count_keywords(text, NEGATIVE_KEYWORDS)
            if pos_c >= neg_c:
                new_sentiment = '긍정'
                sentiment_reason = "격려/위로 맥락에서 부정 키워드 사용 (전체 톤은 긍정)"

    # === 불확실 판정 ===
    # 점수 차이가 작은 경우 (경계선)
    scores = {
        '투자 이야기': inv_score,
        '콘텐츠 반응': cr_score,
        '커뮤니티 소통': comm_score,
        '서비스 이슈': svc_score,
    }
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_scores) >= 2:
        top_topic, top_score = sorted_scores[0]
        second_topic, second_score = sorted_scores[1]
        if top_score > 0 and second_score > 0 and (top_score - second_score) <= 2:
            is_uncertain = True

    # topic이 수정되었고 점수 차이가 크지 않은 경우도 불확실
    if topic_reason and new_topic != current_topic:
        assigned_score = scores.get(new_topic, 0)
        original_score = scores.get(current_topic, 0)
        if assigned_score - original_score <= 3:
            is_uncertain = True

    return new_topic, topic_reason, new_sentiment, sentiment_reason, is_uncertain


# ============================================================
# 4. 실행
# ============================================================

def main():
    # 데이터 로드
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"전수검사 시작: {len(data)}건")
    print("=" * 70)

    # 수정 추적
    topic_changes = []
    sentiment_changes = []
    uncertain_items = []

    reviewed_data = []

    for idx, item in enumerate(data):
        new_topic, topic_reason, new_sentiment, sentiment_reason, is_uncertain = review_item(item)

        reviewed_item = {
            'text': item['text'],
            'original_category': item['original_category'],
            'topic': new_topic,
            'sentiment': new_sentiment,
        }
        reviewed_data.append(reviewed_item)

        # Topic 변경 기록
        if new_topic != item['topic']:
            change = {
                'index': idx,
                'text': item['text'][:80],
                'original_topic': item['topic'],
                'new_topic': new_topic,
                'reason': topic_reason,
            }
            topic_changes.append(change)

        # Sentiment 변경 기록
        if new_sentiment != item['sentiment']:
            change = {
                'index': idx,
                'text': item['text'][:80],
                'original_sentiment': item['sentiment'],
                'new_sentiment': new_sentiment,
                'reason': sentiment_reason,
            }
            sentiment_changes.append(change)

        # 불확실 항목 기록
        if is_uncertain:
            uncertain_item = {
                'index': idx,
                'text': item['text'][:200],
                'original_category': item['original_category'],
                'original_topic': item['topic'],
                'assigned_topic': new_topic,
                'original_sentiment': item['sentiment'],
                'assigned_sentiment': new_sentiment,
                'topic_reason': topic_reason,
                'sentiment_reason': sentiment_reason,
            }
            uncertain_items.append(uncertain_item)

    # === 저장 ===
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(reviewed_data, f, ensure_ascii=False, indent=2)
    print(f"\n수정된 데이터 저장: {OUTPUT_FILE}")

    with open(UNCERTAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(uncertain_items, f, ensure_ascii=False, indent=2)
    print(f"불확실 항목 저장: {UNCERTAIN_FILE} ({len(uncertain_items)}건)")

    # === 통계 출력 ===
    print("\n" + "=" * 70)
    print("[ Topic 수정 통계 ]")
    print(f"총 수정 건수: {len(topic_changes)}")

    # reason별 집계
    topic_reason_counter = Counter()
    for c in topic_changes:
        # 간략화된 reason 키 추출
        reason = c['reason']
        if '→투자' in reason:
            key = f"{c['original_topic']} → 투자 이야기"
        elif '→콘텐츠' in reason:
            key = f"{c['original_topic']} → 콘텐츠 반응"
        elif '→커뮤니티' in reason:
            key = f"{c['original_topic']} → 커뮤니티 소통"
        elif '→서비스' in reason:
            key = f"{c['original_topic']} → 서비스 이슈"
        else:
            key = reason
        topic_reason_counter[key] += 1

    for reason, count in topic_reason_counter.most_common():
        print(f"  {reason}: {count}건")

    print(f"\n[ Sentiment 수정 통계 ]")
    print(f"총 수정 건수: {len(sentiment_changes)}")

    sent_reason_counter = Counter()
    for c in sentiment_changes:
        reason_key = c.get('reason', '기타')
        if reason_key:
            sent_reason_counter[reason_key] += 1

    for reason, count in sent_reason_counter.most_common():
        print(f"  {reason}: {count}건")

    # 전환 방향별 집계
    sent_direction_counter = Counter()
    for c in sentiment_changes:
        key = f"{c['original_sentiment']} → {c['new_sentiment']}"
        sent_direction_counter[key] += 1

    print(f"\n  [방향별]")
    for direction, count in sent_direction_counter.most_common():
        print(f"    {direction}: {count}건")

    # === 수정 전/후 분포 비교 ===
    print(f"\n{'=' * 70}")
    print("[ 수정 전/후 분포 비교 ]")

    # Topic 분포
    before_topics = Counter(d['topic'] for d in data)
    after_topics = Counter(d['topic'] for d in reviewed_data)

    print(f"\n  Topic 분포:")
    print(f"  {'토픽':<15} {'수정 전':>8} {'수정 후':>8} {'변화':>8}")
    print(f"  {'-'*43}")
    all_topics = sorted(set(list(before_topics.keys()) + list(after_topics.keys())))
    for t in all_topics:
        before = before_topics.get(t, 0)
        after = after_topics.get(t, 0)
        diff = after - before
        sign = '+' if diff > 0 else ''
        print(f"  {t:<15} {before:>8} {after:>8} {sign}{diff:>7}")

    # Sentiment 분포
    before_sents = Counter(d['sentiment'] for d in data)
    after_sents = Counter(d['sentiment'] for d in reviewed_data)

    print(f"\n  Sentiment 분포:")
    print(f"  {'감성':<10} {'수정 전':>8} {'수정 후':>8} {'변화':>8}")
    print(f"  {'-'*38}")
    all_sents = sorted(set(list(before_sents.keys()) + list(after_sents.keys())))
    for s in all_sents:
        before = before_sents.get(s, 0)
        after = after_sents.get(s, 0)
        diff = after - before
        sign = '+' if diff > 0 else ''
        print(f"  {s:<10} {before:>8} {after:>8} {sign}{diff:>7}")

    # Cross-tab (수정 후)
    print(f"\n  Topic x Sentiment (수정 후):")
    for t in all_topics:
        for s in all_sents:
            cnt = sum(1 for d in reviewed_data if d['topic'] == t and d['sentiment'] == s)
            if cnt:
                print(f"    {t} x {s}: {cnt}")

    # === 수정 샘플 출력 ===
    print(f"\n{'=' * 70}")
    print("[ Topic 수정 샘플 (최대 30건) ]")
    for i, c in enumerate(topic_changes[:30]):
        print(f"\n  [{i+1}] idx={c['index']}")
        print(f"      텍스트: {c['text']}...")
        print(f"      변경: {c['original_topic']} → {c['new_topic']}")
        print(f"      사유: {c['reason']}")

    if len(topic_changes) > 30:
        print(f"\n  ... 외 {len(topic_changes) - 30}건 추가")

    print(f"\n{'=' * 70}")
    print("[ Sentiment 수정 샘플 (최대 30건) ]")
    for i, c in enumerate(sentiment_changes[:30]):
        print(f"\n  [{i+1}] idx={c['index']}")
        print(f"      텍스트: {c['text']}...")
        print(f"      변경: {c['original_sentiment']} → {c['new_sentiment']}")
        print(f"      사유: {c['reason']}")

    if len(sentiment_changes) > 30:
        print(f"\n  ... 외 {len(sentiment_changes) - 30}건 추가")

    print(f"\n{'=' * 70}")
    print(f"[ 불확실 항목: {len(uncertain_items)}건 ]")
    for i, u in enumerate(uncertain_items[:20]):
        print(f"\n  [{i+1}] idx={u['index']}")
        print(f"      텍스트: {u['text'][:80]}...")
        print(f"      Topic: {u['original_topic']} → {u['assigned_topic']}")
        print(f"      Sentiment: {u['original_sentiment']} → {u['assigned_sentiment']}")
        if u['topic_reason']:
            print(f"      Topic 사유: {u['topic_reason']}")
        if u['sentiment_reason']:
            print(f"      Sentiment 사유: {u['sentiment_reason']}")

    if len(uncertain_items) > 20:
        print(f"\n  ... 외 {len(uncertain_items) - 20}건 추가 (uncertain_items.json 참조)")

    print(f"\n{'=' * 70}")
    print("전수검사 완료.")


if __name__ == '__main__':
    main()
