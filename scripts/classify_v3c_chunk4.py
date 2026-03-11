#!/usr/bin/env python3
"""
Classify chunk_4 items using v3c priority-based taxonomy.

Priority order:
1. 운영 피드백 - Operations: refund, membership enrollment/cancellation, seminar inquiry, shipping, pricing, subscription process complaints
2. 서비스 피드백 - Dev/tech: app crash, payment system error, login failure, feature request, phishing report, broken links
3. 콘텐츠·투자 - BROADEST: any investment/content/master/market mention
4. 일상·감사 - ONLY when P1-P3 have ZERO signals
5. 기타 - Meaningless noise
"""

import json
import re
from collections import Counter

INPUT_PATH = "/Users/gygygygy/Documents/ai/letter-post-weekly-report/data/training_data/v3/relabel_chunks/chunk_4_classified.json"
OUTPUT_PATH = "/Users/gygygygy/Documents/ai/letter-post-weekly-report/data/training_data/v3/relabel_chunks/chunk_4_v3c.json"


def is_noise(text: str) -> bool:
    """Priority 5: meaningless noise - dots, numbers, consonants only, test posts"""
    cleaned = text.strip()
    if len(cleaned) == 0:
        return True
    # Only dots, spaces, special chars, numbers
    if re.fullmatch(r'[.\s·…~!?,\-_*#@\d]+', cleaned):
        return True
    # Only numbers
    if re.fullmatch(r'\d+', cleaned):
        return True
    # "테스트", "test"
    if cleaned.lower().strip() in ['테스트', 'test']:
        return True
    # Only Korean consonants/vowels (no full syllables)
    if len(cleaned) <= 3 and re.fullmatch(r'[ㄱ-ㅎㅏ-ㅣ\s.!?]+', cleaned):
        return True
    return False


def is_operations(text: str) -> bool:
    """Priority 1: 운영 피드백 - operations team handles"""
    t = text.lower()

    # Refund / cancellation / membership process
    refund_patterns = [
        r'환불', r'해지', r'취소', r'결제.*자동', r'자동.*결제',
        r'구독.*취소', r'멤버십.*해지', r'멤버쉽.*해지',
        r'탈퇴', r'결제.*취소',
    ]

    # Enrollment / registration process
    enrollment_patterns = [
        r'등록.*언제', r'언제.*등록', r'신청.*못\s', r'못.*신청',
        r'모집.*언제', r'\d기\s*모집', r'모집.*\d기',
        r'봄학기.*신청', r'겨울학기.*신청',
        r'수강.*신청', r'입학.*신청',
        r'어떻게.*신청', r'신청.*어떻게',
        r'가입.*방법', r'등록.*방법',
    ]

    # Shipping / physical goods
    shipping_patterns = [
        r'택배', r'배송', r'송장', r'도착.*예정',
        r'선물.*보냈', r'보냈.*선물',
    ]

    # Customer service / operations
    ops_patterns = [
        r'고객센터', r'1:1\s*문의', r'고객.*센터',
        r'전화.*연결.*안', r'연결.*안.*전화',
    ]

    # Community management (moderation) - operations handles user management
    moderation_patterns = [
        r'관리.*안\s*되', r'비멤버.*들어와',
        r'멤버십.*아닌.*사람', r'멤버쉽.*아닌.*사람',
        r'맴버쉽.*아닌', r'맴버십.*아닌',
        r'사재기',
    ]

    # Class/membership enrollment inquiry (not content quality)
    class_enrollment = [
        r'모집.*예정', r'모집.*안하', r'모집.*없',
        r'지인.*추천.*모집', r'추천.*모집',
        r'학기.*등록.*언제', r'등록.*학기.*언제',
        r'도감.*어디서.*신청', r'기업분석도감.*신청',
    ]

    # Seminar / live recording request (operational)
    live_patterns = [
        r'녹화본.*올려', r'올려.*녹화본',
    ]

    # Ban / class assignment issues
    admin_patterns = [
        r'개나리반.*아니', r'돌고래반.*아니',
        r'반이.*아닙니다', r'반이.*아니고',
    ]

    all_patterns = (refund_patterns + enrollment_patterns + shipping_patterns +
                    ops_patterns + moderation_patterns + class_enrollment +
                    live_patterns + admin_patterns)

    for pattern in all_patterns:
        if re.search(pattern, t):
            return True

    return False


def is_service_feedback(text: str) -> bool:
    """Priority 2: 서비스 피드백 - dev/tech team fixes"""
    t = text.lower()

    patterns = [
        # App/system bugs
        r'앱.*크래시', r'앱.*오류', r'앱.*에러', r'앱.*안\s*됨',
        r'로그인.*안\s*됨', r'로그인.*안\s*되', r'로그인.*안됩',
        r'접속.*안\s*됨', r'접속.*안\s*되',
        r'결제.*오류', r'결제.*에러', r'결제.*안\s*됨',
        r'결제.*안\s*되',
        # Streaming issues
        r'소리.*작아', r'소리.*안\s*들', r'소리.*끊기',
        r'마이크.*바꿔', r'마이크.*교체',
        r'연결.*상태.*안\s*좋', r'소리.*끊',
        # Login / access issues
        r'로그인이?\s*안', r'접속이?\s*안',
        r'전화번호.*로그인', r'로그인.*전화번호',
        # Feature request (app-level)
        r'알림.*기능', r'new.*없어지지.*않',
        # Phishing / fraud report - warning others about scam channels
        r'피싱', r'사칭', r'사기.*신고',
        r'운영하지\s*않습니다', r'채팅방.*밴드.*운영',
        # Membership system disappeared
        r'멤버십이?\s*사라', r'멤버쉽이?\s*사라',
        r'맴버쉽이?\s*사라', r'맴버십이?\s*사라',
        # Technical: can't access content due to system bug
        r'수강.*안\s*되고\s*있', r'수강이\s*안되고',
        r'뉴스레터.*읽을\s*수.*없',
    ]

    for pattern in patterns:
        if re.search(pattern, t):
            return True

    return False


def has_investment_content_signal(text: str) -> bool:
    """Priority 3: 콘텐츠·투자 - check for ANY investment/content/master/market signal.
    This is the BROADEST category - even one small signal is enough."""
    t = text.lower()

    # === KEYWORD LISTS (substring match) ===

    # Stock names, tickers, companies
    stock_keywords = [
        '삼성전자', '하이닉스', 'sk하이닉스', 'lg전자', 'lg화학', 'lg이노텍',
        '현대차', '대한항공', '두산', '에너빌리티', '한국전력', '한전',
        '카카오', '네이버', '셀트리온', 'hd현대', '포스코',
        '테슬라', 'tesla', '엔비디아', 'nvidia', '애플', 'apple',
        '마이크로소프트', '구글', '아마존', '메타',
        '앱러빈', '로빈후드', '팔란티어', '슈퍼마이크로',
        '샌디스크', 'sandisk', '센트러스', 'centrus',
        '하이브', '에코프로', '포스코퓨처엠',
        'lx인터내셔널', '한조', '대우조선', '한화오션', '한화솔루션',
        '원익', 'qnc', 'asts', '두산밥캣', '두산우',
        'ethu', '비트코인', '이더리움',
        '금값', '은값', '금투자',
        '한온', '한온시스템', '벽산', '브이티', '리게티', '아이온큐', '아이온쿠',
        '오션플랜트', 'sk오션', '대우건설', '디엘이엔씨', 'dl이엔씨', 'dl 이엔씨',
        '삼성에피스', 'hmm', '한화', '에코자매',
        '포스코홀딩스', '에코프로비엠',
        '원전', '풍력', '발전기',
        'vixy', '코스닥150',
    ]

    # Sector/theme keywords
    sector_keywords = [
        '반도체', '2차전지', '이차전지', '석탄',
        '북극항로', '조선', '양자', '양자컴퓨터',
        '인공지능', '바이오', '제약', '방산', '방위산업',
        '전력', '인프라', '송전', '발전', '에너지',
        '자동차', '배터리', '로봇', '드론', '우주',
        '피지컬', '소프트웨어', '클라우드', '데이터센터',
        '관련주', '테마', '섹터', '우주항공',
        '전력동맹', '수혜주', '파워모듈',
        'mro', '국산화',
    ]

    # Investment action keywords
    investment_keywords = [
        '매수', '매도', '분할매수', '분할매도', '손절', '익절',
        '수익', '손실', '수익률', '수익율', '수익율',
        '포트', '포폴', '포트폴리오', '리밸런싱', '리밸',
        '비중', '몰빵', '레버리지', '인버스',
        '주가', '차트', '추세', '지지선', '저항선',
        '목표가', '적정가', '밸류', '밸류에이션',
        '시가총액', '시총', 'per', 'pbr', 'roe', 'eps',
        '실적', '매출', '영업이익', '순이익',
        '배당', '배당금', '배당률',
        '물타', '물렸', '물림', '물타기', '불타기',
        '저점', '고점', '바닥',
        '상승', '하락', '폭락', '폭등', '급락', '급등',
        '매매', '트레이딩', '스윙', '단타', '장투',
        '장기투자', '단기투자', '중장기',
        '들고가', '들고갈', '홀딩', '보유',
        '사고싶', '매수하고', '매도하고',
        '투자금', '시드', '씨드', '원금',
        '존버', '손해',
        '주린이', '주식', '증시',
        '이익실현', '손실확정',
        '전환사채', '공모주', '유상증자', '무상증자',
        '분할.*매', '스위칭',
        '종목', '주도주',
        '마이너스', '플러스',
        '계좌', '평단', '평단가', '본전',
        '하방', '상방',
        '기우제',  # 주식 기우제 = hoping stocks go up
        '오르겠', '오르겟', '떨어지', '떨어질',
        '코인', '크립토', '비트', '이더',
        '모아가', '모아갈',
        '사야', '살까', '팔까', '사면', '팔면',
        '올라', '올랐', '내려', '내렸', '빠지', '빠졌',
        '담았', '담은', '담고',
        '팔아', '팔았',
        '날라가', '날아가',  # stock going up
    ]

    # Market/macro keywords
    market_keywords = [
        '시장', '증시', '코스피', '코스닥', '나스닥', 's&p', 'sp500',
        '다우', '러셀', 'russell', '지수',
        '환율', '달러', '엔화', '위안', '유로',
        '금리', '기준금리', '연준', 'fed',
        '인플레', '디플레', '경기', '경제',
        'gdp', 'cpi', 'ppi', '고용', '실업',
        '트럼프', '관세', '무역', '제재',
        '미국장', '미국 장', '국장', '한국장',
        '중간선거', '대선',
        '지준금', '유동성', '재정',
        'msci', 'etf',
        '국채', '채권', '회사채',
        '소형주', '대형주', '성장주', '가치주',
        '선물', '옵션', '파생',
        '미대법원', '대법원',
        '소매판매', '쇼크',
        '조정',  # market correction
        '변곡점',
        '우크라', '종전', '전쟁',
        '초록불',  # green = stocks up
        '중국', '일본', '인도', '대만',  # country mentions in market context
    ]

    # Content/learning keywords
    content_keywords = [
        '강의', '수업', '공부', '학습', '배움', '배웠',
        '배우고', '공부하', '공부했', '공부중',
        '발표', '분석', '리포트', '보고서',
        '뷰', '인사이트', '관점', '전망',
        '라이브', '브리핑', '런치브리핑',
        '유튜브', '영상', '콘텐츠', '컨텐츠',
        '데일리', '위클리', '주간',
        '레벨', '실전', '입문',
        '학습기업', '씨앗기업', '새싹기업', '애기기업',
        '포트폴리오 문의', '종목 추천', '종목추천',
        '언데', '언제나데이트', '굿모닝',
        '뉴스레터', '주간보고서',
        '소담소담',  # live show name
        '미래전략실',  # show name
        '아침브리핑',
        '거시경제',
    ]

    # Master/teacher names (these are investment teachers - any reference = content context)
    master_names = [
        '찐샘', '찐쌤', '담쌤', '담샘', '두환쌤', '두환샘',
        '돈깡', '돈사부', '미과장', '박두환', '서재형',
        '에릭', '밝은미래', '이읽남',
        'ㅁㄱㅈ',  # abbreviation for 미과장
    ]

    # Master reference patterns
    master_ref_patterns = [
        r'쌤이', r'쌤의', r'쌤께서', r'쌤을', r'쌤~', r'쌤\s',
        r'선생님의', r'선생님~', r'선생님\s',
        r'교수님의', r'교수님께서', r'교수님\s', r'교수님~',
        r'마스터님', r'대장님', r'선장님', r'사부님',
        r'박사님',
    ]

    # Community/educational context
    community_keywords = [
        '급우', '학우', '학우분',
        '학기', '가을학기', '겨울학기', '봄학기', '여름학기',
        '1기', '2기', '3기', '4기',
        '학생', '수강', '멤버십', '멤버쉽', '맴버십', '맴버쉽',
        '어스플러스', '어스캠퍼스', '어스',
        '오피셜클럽',
        '텃밭', '정원',  # community investment terms
        '성적표',
        '홍매화반', '개나리반', '돌고래반',
        '커리큘럼',
        '스터디',
    ]

    # Investment community slang/context
    community_investment = [
        '기업', '섹터', '산업',
        '투자', '투자자', '자산',
        '큰 그림', '큰그림',
        '주주총회', '주총',
        '실적발표', '어닝콜',
        '리딩방', '리딩',
        '댓글알바',  # accusations of shilling = investment community context
        '호위무사',  # defense of master = investment community context
        '뽀큘',
    ]

    all_keywords = (stock_keywords + sector_keywords + investment_keywords +
                    market_keywords + content_keywords + master_names +
                    community_keywords + community_investment)

    for kw in all_keywords:
        if kw in t:
            return True

    # === PATTERN MATCHING ===
    content_patterns = [
        r'잘\s*들었', r'잘\s*봤', r'잘\s*읽',  # consumed content
        r'잘\s*듣', r'잘\s*보',  # consuming content
        r'덕분에.*공부', r'공부.*덕분',
        r'입문.*주식', r'주식.*입문',
        r'수익.*공개', r'계좌.*공개', r'계좌.*인증',
        r'산업.*분석', r'기업.*분석',
        r'학기.*마치', r'학기.*끝',
        r'분할.*매',
        r'방향.*튼',
        r'전고점', r'전저점',
        # Percentage mentions (usually investment returns)
        r'-?\d+\s*%', r'-?\d+프로',
        r'\d+만\s*원', r'\d+천\s*원', r'\d+억',
        # Price mentions
        r'\d+만원', r'\d+천원',
        # Stock price movements
        r'오르기\s*시작', r'떨어지기\s*시작',
        # Master reference patterns
    ] + master_ref_patterns

    for pattern in content_patterns:
        if re.search(pattern, t):
            return True

    # Check for teacher references: 쌤, 선생님, 샘, etc.
    # In this investment education community, referencing the teacher = content context
    teacher_refs_simple = ['쌤', '선생님', '교수님', '박사님', '과장님']
    for ref in teacher_refs_simple:
        if ref in t:
            return True

    # Check: any URL (sharing investment-related links)
    if 'http' in t or 'naver.me' in t or 'youtube' in t:
        return True

    return False


def classify_v3c(item: dict) -> str:
    """Apply priority-based v3c classification."""
    text = item.get('text', '').strip()
    t = text.lower()

    # Priority 5: 기타 - meaningless noise
    if is_noise(text):
        return '기타'

    # Priority 1: 운영 피드백
    if is_operations(text):
        # Exception: if primary content is about investment/content quality complaint
        # alongside cancellation/refund, it should be 콘텐츠·투자
        content_quality_with_cancel = (
            ('해지' in t or '환불' in t or '탈퇴' in t) and
            any(kw in t for kw in ['같은 말', '반복', '시간낭비', '시간 낭비',
                                     '배우고', '배웠', '많은 걸', '투자',
                                     '수익', '포폴', '포트', '종목',
                                     '런치', '영상', '콘텐츠'])
        )
        if content_quality_with_cancel:
            return '콘텐츠·투자'
        return '운영 피드백'

    # Priority 2: 서비스 피드백
    if is_service_feedback(text):
        return '서비스 피드백'

    # Priority 3: 콘텐츠·투자
    if has_investment_content_signal(text):
        return '콘텐츠·투자'

    # === FALLBACK HEURISTICS ===
    # For short ambiguous texts in this investment community,
    # check context clues

    # Community-specific: short frustrated/emotional comments about performance
    # are typically about investment performance
    frustration_investment = [
        '하방', '손절', '깨진', '덜 깨진', '깨졌', '망하',
        '버티', '버텨', '파냐', '팔자', '먹은', '터진',
        '수익', '손실', '손해', '마이너스',
        '오를', '올라', '내려', '떨어',
        '잃', '날라', '날아',
        '대박', '쪽박', '물려', '물린', '물타',
        '체리피킹', '왜곡', '앵무새',
        '나아가', '나아가자',
        '민심', '분탕',
        '넘어지', '무너지', '회복',
    ]
    for kw in frustration_investment:
        if kw in t:
            return '콘텐츠·투자'

    # If it mentions "금" (gold) or "은" (silver) in investment context
    # Need to be careful as 은 is also a common Korean particle
    # Only match 금/은 when specifically about precious metals
    gold_silver_patterns = [
        r'금\s*가격', r'은\s*가격', r'금\s*시세', r'은\s*시세',
        r'실물\s*은', r'실물\s*금',
    ]
    for pattern in gold_silver_patterns:
        if re.search(pattern, t):
            return '콘텐츠·투자'

    # Very short comments that are clearly about community drama/investment frustration
    # (masterName context: if directed at investment master, likely investment-related)
    master_name = item.get('masterName', '')
    if master_name and len(text) < 60:
        # Check if it seems to be a reaction to master's investment advice
        reaction_words = ['기도문', '멘트', '앞으로', '강하게', '자신', '비틀',
                         '써봐야', '됐다', '잘고름', '충분', '사람잡', '정치인',
                         '출신', '경력', '달달', '삭제', '지우', '글',
                         '잘 안나', '답답', '쏘는', '예민', '진정',
                         '장난', '우리', '같이']
        for w in reaction_words:
            if w in t:
                return '콘텐츠·투자'

    # Priority 4: 일상·감사 - only when NO signals from above
    return '일상·감사'


def apply_manual_overrides(data: list) -> list:
    """Apply manual overrides for specific items that the rule-based classifier
    cannot handle well due to ambiguity or unusual phrasing."""
    overrides = {
        # "싸이클" = economic/market cycle -> 콘텐츠·투자
        4095: '콘텐츠·투자',
        # "한 치 앞만 보지 말고 조금만 더 멀리 봅시다" - encouraging about market -> 콘텐츠·투자
        4114: '콘텐츠·투자',
        # "매일 올려주시는 덕분에 많은 도움" - reacting to master's content -> 콘텐츠·투자
        4163: '콘텐츠·투자',
        # "1개월 오늘 만기일인데 6개월로 바꾸는 거" - membership duration change -> 운영 피드백
        4601: '운영 피드백',
        # "매월 초 그달의 예상 일정을 올려주실 것을 제안" - content schedule request -> 콘텐츠·투자
        4682: '콘텐츠·투자',
        # "험담을 퍼붓고 욕을 하는건 아동학대" - defending master in community drama -> 콘텐츠·투자
        4807: '콘텐츠·투자',
        # "여기 난리치는 사람들" - community drama about investment losses -> 콘텐츠·투자
        4822: '콘텐츠·투자',
        # "AI 말 안 듣고 대듭니다" - AI-related content discussion -> 콘텐츠·투자
        4844: '콘텐츠·투자',
        # "정신병자들 대가리 속" - toxic noise -> 기타
        4859: '기타',
        # "금지사항입니다" - community rules about content sharing -> 콘텐츠·투자
        4875: '콘텐츠·투자',
        # "스파링 성사 될 경우" - physical fight arrangement, unrelated -> 기타
        4942: '기타',
        # "전화해서 장소 시간 정해라" - physical fight arrangement -> 기타
        4943: '기타',
        # "죽을 맛인 만큼 신고랑 죽일 각오로 덤비겠습니다" - investment frustration -> 콘텐츠·투자
        4990: '콘텐츠·투자',
        # "새로운 모델을 만들어 주세요" - asking master to continue work -> 콘텐츠·투자
        4051: '콘텐츠·투자',
        # "세계의 변화를 알게 해 주셔서 감사" - learning about global changes = content reaction -> 콘텐츠·투자
        4505: '콘텐츠·투자',
        # "무통장 입금 해주셔서 듣고있었는데" - payment method inquiry -> 운영 피드백
        4911: '운영 피드백',
        # "힘든 장이지만 딱 7배 불리는" - investment wish -> 콘텐츠·투자
        4970: '콘텐츠·투자',
        # "근데 사실 이미 다 앎" - community context, ambiguous -> 기타
        4974: '기타',

        # Fix 운영 피드백 items that should be 콘텐츠·투자:
        # "언제나 데이트를 보며... 주식 판에서" - content reaction -> 콘텐츠·투자
        4713: '콘텐츠·투자',
        # "담샘의 음성을 듣고 공부하는 가운데" - content reaction -> 콘텐츠·투자
        4747: '콘텐츠·투자',
        # "2차전지로 5천정도를 번사람... 봄학기 신청" - investment story with enrollment mention -> 콘텐츠·투자
        4775: '콘텐츠·투자',
        # "사실이 아닙니다... 오픈톡방을 만들었고" - community dispute -> 콘텐츠·투자
        4347: '콘텐츠·투자',
        # "수강신청도 하기로 했고 주식도 다시 매입" - investment + enrollment -> 콘텐츠·투자
        4360: '콘텐츠·투자',
        # "셀트리온 조금 더 담을라다가" - investment action with accidental purchase -> 콘텐츠·투자
        4633: '콘텐츠·투자',
        # "유투브로 2년동안 담쌤 구독... 기업도감" - content reaction -> 콘텐츠·투자
        4693: '콘텐츠·투자',
        # "봄학기 강의 신청하면 자동으로 전과집이 배송" - educational material -> 운영 피드백
        4695: '운영 피드백',
        # "우체국택배에서 날아온 문자" - shipping/gift -> 일상·감사
        4721: '일상·감사',
        # "안동농부님... 국수 삶아" - daily life / food, but in context of master's class -> 일상·감사
        4729: '일상·감사',
        # "2026년 봄 학기가 2월 23일부터 시작" - schedule info with investment mention -> 콘텐츠·투자
        4765: '콘텐츠·투자',
        # "버틸거면 군말없이 버티고... 해지하고 쳐 나가면 되지... 관리 안하냐" - community moderation + investment context -> 운영 피드백
        4947: '운영 피드백',
        # "보이스피싱... 무자격 투자자문업자에게 사기 당한" - investment fraud criticism -> 콘텐츠·투자
        4877: '콘텐츠·투자',
        # "한조에 대해 딥하게 공부... 후판 가격협상... 박스권" - pure investment analysis -> 콘텐츠·투자
        4579: '콘텐츠·투자',
        # "멤버십 비용 환불해주고" - pure refund request -> 운영 피드백 (correct)
        # 4803: already correct
        # "라이브는 모두 참여할 수 있도록" - requesting open access for live -> 운영 피드백 (correct)
        # 4312: check if correctly classified
    }

    for item in data:
        if item['idx'] in overrides:
            item['v3c_topic'] = overrides[item['idx']]

    return data


def main():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items")

    # Classify each item
    for item in data:
        item['v3c_topic'] = classify_v3c(item)

    # Apply manual overrides for edge cases
    data = apply_manual_overrides(data)

    # Print distribution
    dist = Counter(item['v3c_topic'] for item in data)
    print("\n=== v3c_topic Distribution ===")
    for topic, count in dist.most_common():
        pct = count / len(data) * 100
        print(f"  {topic}: {count} ({pct:.1f}%)")

    # Compare with v3b
    print("\n=== v3b -> v3c Migration ===")
    migration = Counter()
    for item in data:
        key = f"{item['v3b_topic']} -> {item['v3c_topic']}"
        migration[key] += 1
    for key, count in sorted(migration.items()):
        print(f"  {key}: {count}")

    # Show items that went from 콘텐츠·투자 to 일상·감사 (most suspicious)
    print("\n=== 콘텐츠·투자 -> 일상·감사 (review these) ===")
    suspect = [item for item in data if item['v3b_topic'] == '콘텐츠·투자' and item['v3c_topic'] == '일상·감사']
    print(f"Count: {len(suspect)}")
    for item in suspect:
        text = item['text'][:150].replace('\n', ' ')
        print(f"  [{item['idx']}] {text}")

    # Show all 일상·감사 items
    print("\n=== All 일상·감사 items ===")
    ilsang = [item for item in data if item['v3c_topic'] == '일상·감사']
    print(f"Count: {len(ilsang)}")
    for item in ilsang:
        text = item['text'][:120].replace('\n', ' ')
        print(f"  [{item['idx']}] v3b={item['v3b_topic']}: {text}")

    # Show all 기타 items
    print("\n=== All 기타 items ===")
    gita = [item for item in data if item['v3c_topic'] == '기타']
    print(f"Count: {len(gita)}")
    for item in gita:
        text = item['text'][:120].replace('\n', ' ')
        print(f"  [{item['idx']}] v3b={item['v3b_topic']}: {text}")

    # Save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
