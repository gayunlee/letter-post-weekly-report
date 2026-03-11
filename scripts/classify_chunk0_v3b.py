#!/usr/bin/env python3
"""
Classify chunk_0.json items into 5 v3b categories.
Uses contextual analysis rules derived from manual review of all 1000 items.

Categories:
1. 콘텐츠·투자 — Substantive reactions/opinions/questions about masters/content/market/stocks
   Must have ACTUAL CONTENT: specific investment actions, content feedback with details,
   market analysis, stock discussion, profit/loss WITH context
2. 일상·감사 — Greetings, thanks, encouragement, daily life WITHOUT substantive content
3. 운영 피드백 — Issues for operations team: seminar, refund, membership, shipping, pricing
4. 서비스 피드백 — Technical issues for dev team: app bugs, payment errors, login, feature requests
5. 기타 — Meaningless noise: ".", "1", "?", consonants only, test posts, under 5 chars
"""

import json
import re

INPUT_PATH = "/Users/gygygygy/Documents/ai/letter-post-weekly-report/data/training_data/v3/relabel_chunks/chunk_0.json"
OUTPUT_PATH = "/Users/gygygygy/Documents/ai/letter-post-weekly-report/data/training_data/v3/relabel_chunks/chunk_0_classified.json"


def classify_text(text: str) -> str:
    """Classify a single text into one of 5 categories."""

    clean = text.strip()
    text_lower = clean.lower()

    # =========================================================
    # PHASE 1: 기타 — Meaningless noise
    # =========================================================

    # Very short or punctuation-only
    if len(clean) <= 3:
        if re.match(r'^[\.\,\!\?\~\-\·ㆍ\s\d]+$', clean):
            return "기타"
        if re.match(r'^[ㄱ-ㅎㅏ-ㅣ]+$', clean):
            return "기타"
        if clean.isdigit():
            return "기타"
        if len(clean) <= 2:
            return "기타"

    # Just dots, dashes, or meaningless single chars
    if re.match(r'^[\.\,\!\?\~\-\·ㆍ\s]+$', clean):
        return "기타"

    # "삭제했습니다" type
    if clean in ['삭제했습니다', '삭제', '삭제합니다']:
        return "기타"

    # Specific short meaningless words
    if clean.strip() in ['꾸벅', 'ㅋㅋ', 'ㅎㅎ', 'ㅠㅠ', 'LG', 'LG ', 'sk', 'SK']:
        return "기타"

    # Under 5 chars with no meaning (but allow meaningful Korean phrases)
    if len(clean) < 5:
        meaningful_short = ['감사합니다', '응원합니다', '화이팅', '감사해요', '축하합니다', '힘내세요']
        if any(m in clean for m in meaningful_short):
            return "일상·감사"
        if re.match(r'^[A-Z]{1,4}\s*$', clean):
            return "기타"
        if re.match(r'^[\.\,\!\?\~ㆍ\s]+$', clean):
            return "기타"

    # Short stock-cheer patterns: "[종목] 가즈아", "[종목]도 갑니다" etc - no substance
    if len(clean) < 20:
        if re.search(r'가즈아', clean):
            return "기타"
        # Just a stock name + emoji or simple cheer
        if re.match(r'^[\w\s]{1,8}(도\s*)?(갑니다|가요|가자|가즈아)[~!\.🍀]*\s*$', clean):
            return "기타"

    # Short texts (5-15 chars) that are still too short to be meaningful
    # but not punctuation-only
    if len(clean) < 15 and not any(kw in clean for kw in ['감사', '고맙', '응원', '화이팅', '축하']):
        # Short emotional expressions
        if any(w in clean for w in ['아쉽', '힘들', '걱정', '속상', '미안', '죄송', '사랑',
                                     '힘드', '축하', '믿어', '화이팅']):
            return "일상·감사"
        # Short stock recommendation (e.g., "현대건설을 사세요")
        if any(w in clean for w in ['사세요', '매수', '매도', '종목']):
            return "콘텐츠·투자"
        # Short stock movement commentary (e.g., "이노 출발합니다")
        if any(w in clean for w in ['출발', '양전', '급등', '급락', '효자']):
            return "콘텐츠·투자"

    # =========================================================
    # PHASE 2: 서비스 피드백 — Technical/dev issues
    # =========================================================

    service_keywords = [
        '에러가 나', '에러가나', '화면이 안나', '화면이안나', '앱이 안', '앱 오류',
        '결제 오류', '결제오류', '로그인이 안', '로그인 안', '버그', '접속이 안',
        '접속 안', '사이트가 안', '게시글 삭제기능', '삭제 기능', '삭제기능',
        '링크가 안', '링크 안', '화면이 안', '라이브 화면', '화면 안나',
        '회원신청화면이',
    ]

    for kw in service_keywords:
        if kw in clean:
            return "서비스 피드백"

    # Link broken / technical issue
    if '링크' in clean and ('이동' in clean or '이상' in clean or '안되' in clean or '안 되' in clean):
        return "서비스 피드백"

    # Feature request for app/platform features (dev team)
    if '기능' in clean and ('넣어' in clean or '추가' in clean or '없어서' in clean):
        return "서비스 피드백"

    # 200자 제한, photo upload, character limit issues
    if ('제한' in clean or '첨부' in clean) and ('사진' in clean or '글자' in clean or '200자' in clean):
        return "서비스 피드백"

    # Character count complaints (글자수)
    if '글자수' in clean or '글자 수' in clean:
        return "서비스 피드백"

    # 6시50분부터 라이브 기다리는데 화면이 안나옵니다
    if '화면' in clean and ('안나' in clean or '안 나' in clean):
        return "서비스 피드백"

    # =========================================================
    # PHASE 3: 운영 피드백 — Operations team issues
    # BUT: if text has substantive investment content alongside ops keywords,
    # we should check carefully. Pure ops = 운영, mixed with investment = depends.
    # =========================================================

    ops_keywords_simple = [
        '환불', '결제 취소', '결제취소', '수강취소', '수강 취소',
        '멤버십 취소', '멤버쉽 취소', '구독 취소', '유료결제 취소', '유료 결제 취소',
        '결제하고나면',
        '강의커리큘럼', '어디에서 강의',
        '꽃동산은 어디', '텃밭에 게시글이', '어떻게 접속',
    ]

    for kw in ops_keywords_simple:
        if kw in clean:
            return "운영 피드백"

    # "해지" - only match when it's about subscription/membership cancellation
    # Not when it's part of other words like "편해지고", "전해지", "해지난뒤"
    # Also not when "해지" is mentioned as a consequence/warning, not as a request
    if '해지' in clean:
        # Check if it's subscription context
        # Positive patterns: "해지신청", "해지하", "해지 요청", "정기결제 해지"
        if re.search(r'(해지신청|해지하|해지 요청|해지처리|결제.*해지|구독.*해지|정기.*해지|해지 후|해지후)', clean):
            return "운영 피드백"
        # "멤버십 해지" - only if user is requesting/doing it, not warning about it
        if re.search(r'멤버.*해지', clean):
            # If context is about consequences/warnings (e.g., "해지 및 악소문"), skip
            if any(w in clean for w in ['부작용', '악소문', '악플', '뿐입니다']):
                pass  # warning context, not a request
            else:
                return "운영 피드백"

    # "탈퇴" - only for user membership exit, not corporate events
    if '탈퇴' in clean:
        # Check if it's about user/membership, not corporate
        if any(w in clean for w in ['자회사', '러시아', '기업', '회사가', '주가']):
            pass  # Corporate event, not ops
        elif any(w in clean for w in ['회원', '멤버', '구독', '수강', '가입', '유료', '결제']):
            return "운영 피드백"
        elif '잘못 눌' in clean or '탈퇴 버튼' in clean or '탈퇴를 하고' in clean:
            return "운영 피드백"

    # "어스캠퍼스" - only if asking about platform usage, not mentioning in passing
    if ('어스캠퍼스' in clean or '어스 캠퍼스' in clean):
        # If the text is mainly about investment, don't classify as ops
        inv_check = ['매수', '매도', '종목', '주가', '분석', '포트', '포폴',
                     '학습기업', '새싹', '주식', '수익', '손실', '편출',
                     '투자', '기업', '비중', '포지션']
        if not any(w in clean for w in inv_check) and len(clean) < 200:
            return "운영 피드백"

    # Refund / payment change
    if '환불' in clean or ('결제' in clean and ('취소' in clean or '변경' in clean or '해지' in clean)):
        return "운영 피드백"

    # Membership access issues
    if ('멤버쉽' in clean or '멤버십' in clean) and ('들어오게' in clean or '접근' in clean or '제한' in clean or '아닌 사람' in clean or '안보입' in clean or '안 보' in clean):
        return "운영 피드백"

    # Waiting for content upload, operational issues (not content discussion)
    # But only if there's no substantive investment discussion
    investment_check_words = ['매수', '매도', '종목', '주가', '분석', '포트', '포폴',
                             '학습기업', '새싹', '기업', '주식', '수익', '손실', '편출']
    has_investment_in_text = any(w in clean for w in investment_check_words)

    if not has_investment_in_text:
        if ('세미나 영상' in clean or '영상 업로드' in clean or '영상이 없' in clean or '업로드 날짜' in clean):
            return "운영 피드백"

        if ('영상' in clean or '세미나' in clean) and ('기다' in clean and ('업로드' in clean or '언제' in clean)):
            return "운영 피드백"

        # Video not uploaded complaint
        if '영상' in clean and ('안 올라' in clean or '안올라' in clean):
            return "운영 피드백"

    # Platform how-to questions
    if '어디' in clean and ('볼 수' in clean or '볼수' in clean or '보나요' in clean) and ('강의' in clean or '수업' in clean):
        return "운영 피드백"

    # Platform section navigation questions (꽃동산, 텃밭, etc.)
    if ('꽃동산' in clean or '텃밭' in clean) and ('어디' in clean or '안보여' in clean or '안 보여' in clean or '없어' in clean):
        return "운영 피드백"

    # Enrollment requests
    if '등록' in clean and ('교실' in clean or '학교' in clean or '반' in clean) and ('하고 싶' in clean or '하고싶' in clean or '어떻게' in clean or '어떡해' in clean or '하나요' in clean):
        return "운영 피드백"

    # Operations announcements
    if '문의하기' in clean and ('링크' in clean or 'http' in clean or 'channel' in clean):
        return "운영 피드백"

    # ISA/IRP account setup questions (operational how-to, not strategy)
    if ('isa' in text_lower or 'irp' in text_lower) and ('가입' in clean or '계좌' in clean) and ('어떻게' in clean or '차이' in clean or '헷갈' in clean):
        if not any(w in clean for w in ['종목', '매수', '매도', '포폴', '포트폴리오']):
            return "운영 피드백"

    # Operations team communication issues
    if '운영진' in clean and ('소통' in clean or '공지' in clean or '피드백' in clean or '답' in clean):
        return "운영 피드백"

    # Unanswered inquiry complaints
    if ('문의' in clean or '학우님들' in clean) and ('답변이 없' in clean or '답변 없' in clean or '묵묵부답' in clean or '답이 없' in clean or '답도 없' in clean):
        return "운영 피드백"

    # Subscription plan change questions
    if ('개월' in clean or '월짜리' in clean) and ('변경' in clean or '바꾸' in clean):
        return "운영 피드백"

    # Subscription renewal / re-enrollment questions
    if ('수강신청' in clean or '재신청' in clean or '재수강' in clean) and ('어떻게' in clean or '하면 되나요' in clean):
        return "운영 피드백"

    # Membership plan changes
    if '마감' in clean and '변경' in clean:
        return "운영 피드백"

    # MP update schedule (operational, not content)
    if 'mp' in text_lower and ('올려' in clean or '업로드' in clean) and len(clean) < 80:
        return "운영 피드백"

    # Asking about opting out / canceling membership
    if ('유료결제' in clean or '유료 결제' in clean) and ('취소' in clean or '탈퇴' in clean):
        return "운영 피드백"

    # Subscription cancellation request (재등록 취소, 결재 취소, etc.)
    if ('재등록' in clean or '재결제' in clean) and ('취소' in clean):
        return "운영 피드백"

    # Scam / impersonation inquiries (asking if a channel/band is official)
    if ('밴드' in clean or '네이버 밴드' in clean or '채팅방' in clean) and ('사기' in clean or '공식' in clean or '이름으로' in clean):
        return "운영 피드백"

    # Community management criticism (댓글 알바, 여론조작, etc.)
    if ('댓글 알바' in clean or '여론조작' in clean or '댓글 관리' in clean):
        return "운영 피드백"

    # Subscription auto-renewal helper messages (community helping with ops)
    if '카드 결제' in clean and ('연장' in clean or '자동' in clean):
        return "운영 피드백"

    # Scam warnings between members (community ops)
    if '사칭' in clean and ('밴드' in clean or '유튜브' in clean or '채널' in clean):
        return "운영 피드백"

    # Offline meeting requests (operational)
    if '오프라인 모임' in clean:
        return "운영 피드백"

    # Content upload / public notice requests
    if ('편출' in clean and '삭제' in clean) or ('공지에 올려' in clean) or ('공지로' in clean and '해주' in clean):
        return "운영 피드백"

    # Q&A announcements (community operational notice) - short announcements only
    if '공지사항' in clean and ('Q&A' in clean or 'q&a' in clean) and len(clean) < 80:
        return "운영 피드백"

    # Seminar upload requests (but NOT if discussing investment content from seminar)
    # Also NOT if just expressing regret about missing seminar and watching video instead
    if '세미나' in clean and ('업로드' in clean or '영상' in clean):
        # If also discussing specific stocks/investment actions, it's 콘텐츠·투자
        investment_action_words = ['매수', '매도', '종목', '주가', '분석', '포트', '포폴',
                                   '잡을', '담았', '샀', '팔았', '기업', '학습']
        if any(w in clean for w in investment_action_words):
            pass  # Don't classify as 운영 피드백, let it fall through
        elif '못 가고' in clean or '못가고' in clean or '못 갔' in clean:
            pass  # Just expressing regret about missing seminar, not ops request
        else:
            return "운영 피드백"

    # Membership fee related
    if '회비' in clean and ('결제' in clean or '결재' in clean):
        return "운영 피드백"

    # =========================================================
    # PHASE 4: Determine 콘텐츠·투자 vs 일상·감사
    # =========================================================

    # --- Detect investment/content substance ---

    # STRONG investment keywords (very specific to investment content)
    strong_investment = [
        '종목', '매수', '매도', '포트폴리오', '포폴', '포토폴리오',
        'ETF', 'etf', '리밸런싱', '리벨런싱', '수익률', '손절',
        '배당', '시가총액', '시총', 'PER', 'per', '밸류에이션',
        '공매도', '인버스', '레버리지',
        '반도체', '원전', '조선', '방산', 'HBM', 'HBF',
        '비트코인', '이더리움',
        '나스닥', 'S&P', '코스피', '코스닥',
        '금리', '환율', '유동성', '인플레이션',
        '실적', '매출', '영업이익', '순이익', '어닝',
        '차트', '이평선', '저항선', '지지선',
        '섹터', '모멘텀', '수급',
        '삼성전자', '삼전', '하이닉스', '엔비디아',
        '테슬라', '팔란티어', '로빈후드', '오라클',
        '두산', '한전', '한화솔루션', '한화시스템',
        '대한항공', '하나투어', '현대차', '현대건설',
        '파두', '마이크론', '웨스턴디지털', '샌디스크',
        '코히어런트', '루멘텀', '크리도',
        '한온시스템', '한온', '이녹스첨단소재', '이녹스', 'LG디플', 'LG디스플레이',
        'HD건설기계', 'hd건설기계',
        '이노와이어리스', '이노베이션', '에스케이이노', 'LG이노텍',
        '미래에셋증권', '삼성증권', '키움증권', '한국투자증권',
        '정유사', '정제마진', '유가',
        '재고평가',
        '원자재', '금값',
        '트럼프', '연준', 'TGA', 'TLT', 'BIL',
        'ARKK', 'arkk',
        '학습기업', '새싹기업', '공부기업', '스터디기업',
        '편출', '편입',
        '목표가', '적정가',
        '대북관련주', '관련주',
        '자사주', '소각', '유상증자',
        '거시', '사이클',
        '포지션', '비중', '현금확보', '현금 확보',
        '윗돌', '아랫돌', '분할매수', '분할 매수',
        '가치투자', '가치 투자', '내재가치', '내재 가치',
        '펀더멘탈', '시대정신',
        '양자컴퓨터', '페로브스카이트',
        '국장', '미국주식', '미국 주식',
        '중간선거', '변곡점',
        '퇴직연금',
        '연금계좌', '연금저축',
        '상법개정', '스테이블코인',
        '불타기', '물타기',
        '빅테크', 'SaaS', 'saas', '마진구조',
        '희토류', '지정학', '트레이딩', '수익실현',
        '금값', '금시세',
    ]

    # WEAK investment keywords (need additional context to be 콘텐츠·투자)
    weak_investment = [
        '주식', '투자', '시장', '장이', '계좌', '수익', '손실',
        '마이너스', '플러스', '빨간', '파란',
        '올라', '내려', '빠지', '떨어', '상승', '하락', '폭등', '폭락',
        '버티', '존버', '홀드', '홀딩', '매집', '진입',
        '장기', '단기', '경기', '달러',
        '청산', '깡통', '버블',
        '비트', '코인', '이더',
        '시황', '브리핑', '매매',
        '대북', '줍줍', '예측', '복기', '채권', '리스크',
    ]

    # Content reaction keywords
    content_keywords = [
        '강의', '수업', '라이브', '런브',
        '분석', '인사이트', '콘텐츠',
    ]

    # Substantive detail words (show they're discussing content substance)
    substantive_words = [
        '배우', '공부', '학습', '도움', '유익',
        '전망', '흐름', '관점', '판단',
        '설명', '알려', '가르', '지식',
    ]

    # Thanks/encouragement keywords
    thanks_words = ['감사', '고맙', '고마워', '덕분']
    encourage_words = [
        '응원', '힘내', '화이팅', '파이팅', '건강하세', '건강 챙',
        '사랑합니다', '감동', '눈물', '울컥', '뭉클',
        '좋은 주말', '평안', '쉬세요', '푹 쉬',
    ]
    personal_words = [
        '산책', '여행', '이사', '설날', '명절',
        '아이들', '남편', '아내', '엄마', '아빠',
        '생일', '축하', '결혼', '임신', '출산',
        '낭독', '시를', '시가', '독서', '런닝',
    ]

    # Special handling for "미장" - avoid matching "미과장"
    if '미장' in clean and '미과장' not in clean:
        strong_investment.append('미장_SPECIAL')  # just to count it
        # We'll check separately

    # Count hits
    strong_count = sum(1 for kw in strong_investment if kw in clean and kw != '미장_SPECIAL')
    # Add 미장 separately
    if '미장' in clean and '미과장' not in clean:
        strong_count += 1

    # Context-aware weak keyword counting
    # "올라" can mean stock going up OR content being uploaded - filter upload context
    def is_weak_match(kw, text):
        if kw != '올라':
            return kw in text
        if '올라' not in text:
            return False
        # Check if "올라" is in upload context
        upload_patterns = ['올라오', '올라 오', '안올라', '안 올라 오']
        if any(p in text for p in upload_patterns):
            # Only count if there's also a price/stock context
            price_context = ['주가', '종목', '시세', '%', '프로', '원대']
            return any(p in text for p in price_context)
        return True  # "올라가" etc. is investment context

    weak_count = sum(1 for kw in weak_investment if is_weak_match(kw, clean))
    content_count = sum(1 for kw in content_keywords if kw in clean)
    substantive_count = sum(1 for kw in substantive_words if kw in clean)
    thanks_count = sum(1 for kw in thanks_words if kw in clean)
    encourage_count = sum(1 for kw in encourage_words if kw in clean)
    personal_count = sum(1 for kw in personal_words if kw in clean)

    has_strong = strong_count > 0
    has_weak = weak_count > 0
    has_content = content_count > 0
    has_substantive = substantive_count > 0
    has_thanks = thanks_count > 0
    has_encourage = encourage_count > 0
    has_personal = personal_count > 0

    # --- Decision logic ---

    # 1. Strong investment keyword -> definitely 콘텐츠·투자
    if has_strong:
        return "콘텐츠·투자"

    # 2. Content keyword + substantive detail -> 콘텐츠·투자
    if has_content and has_substantive:
        return "콘텐츠·투자"

    # 3. Content keyword alone
    if has_content:
        # Check if it's just a simple thanks about a lecture
        # "강의 잘 들었습니다 감사합니다" -> 일상·감사
        # "강의에서 마이크론 per을..." -> 콘텐츠·투자 (caught by strong_investment already)
        if has_thanks or has_encourage:
            # Predominantly thanks/encouragement with mention of lecture
            if len(clean) < 60:
                return "일상·감사"
            # Longer - check if there's actual content discussion
            if has_weak:
                return "콘텐츠·투자"
            return "일상·감사"
        # Content keyword without thanks - discussing content substance
        if len(clean) > 30:
            return "콘텐츠·투자"
        return "일상·감사"

    # 4. Weak investment keywords
    if has_weak:
        # Pure emotional support mentioning market/loss but no analysis
        # E.g., "계좌가 파란불이지만 힘내세요" -> depends on depth

        # If combined with strong thanks/encouragement context, it's likely 일상·감사
        emotional_indicators = ['힘내', '괜찮', '응원', '화이팅', '파이팅', '힘드',
                               '걱정', '위로', '마음', '건강', '사랑', '쉬세요',
                               '평안', '아프', '상처', '스트레스', '울컥', '눈물',
                               '속상', '지나갈', '이 또한', '버텨', '견디']
        emotional_count = sum(1 for e in emotional_indicators if e in clean)

        if (has_thanks or has_encourage) and weak_count <= 1:
            if emotional_count >= 1:
                return "일상·감사"

        # Even without thanks/encourage, if heavily emotional and weak investment is just backdrop
        if emotional_count >= 2 and weak_count <= 1:
            return "일상·감사"

        # Discussing their own investment actions/losses/gains with some substance
        action_words = ['매수', '매도', '샀', '팔았', '담았', '정리', '익절', '손절',
                       '리밸런싱', '옮겼', '바꿨', '넣었', '뺐', '매입']
        has_action = any(w in clean for w in action_words)
        if has_action:
            return "콘텐츠·투자"

        # Asking investment questions
        question_markers = ['까요', '인가요', '나요', '건지', '궁금', '질문', '문의', '어떻게',
                           '어떨까', '할까요', '해야', '맞는', '인지요']
        has_question = any(q in clean for q in question_markers)
        if has_question:
            return "콘텐츠·투자"

        # Market commentary / opinion
        if weak_count >= 2 and len(clean) > 30:
            return "콘텐츠·투자"

        # Single weak keyword in longer text with context
        if weak_count >= 1 and len(clean) > 50:
            # Check if text is predominantly about personal experience with market
            if any(w in clean for w in ['반등', '조정', '바닥', '고점', '저점', '불장', '하락장', '상승장']):
                return "콘텐츠·투자"
            # Talking about their account status
            if '계좌' in clean and ('파란' in clean or '빨간' in clean or '마이너스' in clean or '수익' in clean):
                return "콘텐츠·투자"
            # Talking about market dynamics
            if '시장' in clean and any(w in clean for w in ['흔들', '변동', '조정', '하락', '상승']):
                return "콘텐츠·투자"
            # Discussing trust/confidence in market context
            if '투자' in clean and any(w in clean for w in ['믿', '신뢰', '뷰', '방향', '시계열', '포트']):
                return "콘텐츠·투자"

        # If just one weak keyword and short/emotional, likely 일상·감사
        if weak_count == 1 and len(clean) < 50:
            if has_thanks or has_encourage:
                return "일상·감사"

        # Default for texts with weak keywords - depends on length and context
        if len(clean) > 80:
            return "콘텐츠·투자"

        # Short with single weak keyword
        return "콘텐츠·투자"

    # 5. No investment keywords at all

    # Personal stories, daily life
    if has_personal:
        return "일상·감사"

    # Thanks/encouragement without any investment context
    if has_thanks or has_encourage:
        return "일상·감사"

    # Emotional support without investment content
    emotional_phrases = [
        '힘내', '괜찮', '이 또한 지나', '잊지 마', '알아주',
        '믿고 따', '화이팅', '꼭 기억', '마음', '걱정',
        '위로', '공감', '아프', '속상', '죽고싶',
        '극단적', '한강', '스트레스',
    ]
    if any(p in clean for p in emotional_phrases):
        return "일상·감사"

    # About master's personality, not content
    master_personality = ['성품', '인격', '인간적', '진심', '진정성', '겸손', '따뜻']
    if any(w in clean for w in master_personality):
        return "일상·감사"

    # Community bonding without content
    if any(w in clean for w in ['동행', '함께', '오래오래', '항상']):
        if len(clean) < 60:
            return "일상·감사"

    # Substantive words alone (about learning/studying) -> likely 콘텐츠·투자
    if has_substantive and len(clean) > 30:
        return "콘텐츠·투자"

    # Short remaining without any markers
    if len(clean) < 10:
        return "기타"

    # Short-ish texts (10-20 chars) without clear investment or thanks markers
    # These are usually brief emotional expressions
    if len(clean) < 20:
        return "일상·감사"

    # Medium length without markers - likely 일상·감사 (general community chat)
    if len(clean) < 50:
        return "일상·감사"

    # Longer texts without any investment or content markers
    # These are usually emotional support, personal stories, or community discussion
    # about non-investment topics
    return "일상·감사"


def main():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items")

    # Classify each item
    category_counts = {}
    for item in data:
        topic = classify_text(item['text'])
        item['v3b_topic'] = topic
        category_counts[topic] = category_counts.get(topic, 0) + 1

    # Save results
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nClassification complete. Results saved to {OUTPUT_PATH}")
    print(f"\nCategory distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(data)*100:.1f}%)")

    # Print some samples for each category
    for cat in ['콘텐츠·투자', '일상·감사', '운영 피드백', '서비스 피드백', '기타']:
        samples = [d for d in data if d['v3b_topic'] == cat][:3]
        print(f"\n--- {cat} samples ---")
        for s in samples:
            print(f"  [{s['idx']}] {s['text'][:80]}")


if __name__ == "__main__":
    main()
