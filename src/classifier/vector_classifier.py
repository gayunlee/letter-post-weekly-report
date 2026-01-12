"""벡터 유사도 기반 콘텐츠 분류 시스템 (하이브리드)"""
import os
from typing import List, Dict, Any, Tuple
from anthropic import Anthropic
from dotenv import load_dotenv
from ..vectorstore.chroma_store import ChromaVectorStore
from .content_classifier import ContentClassifier

load_dotenv()


class VectorContentClassifier:
    """벡터 유사도 기반 콘텐츠 분류기 (빠른 분류)"""

    # 카테고리 분류 기준 (7개):
    # - 감사·후기: 감사 표현, 긍정적 투자 후기
    # - 질문·토론: 투자/종목/비중/전략 질문, 시장 의견 (감정 라벨: 긍정/부정/중립)
    # - 정보성 글: 시장/종목 정보 공유
    # - 서비스 문의: 서비스 이용 방법, 멤버십, 결제 관련 문의 (플랫폼 운영)
    # - 서비스 불편: 플랫폼/앱 버그, 서비스 운영 불만 (플랫폼 운영)
    # - 서비스 제보/건의: 사칭 제보, 기능 요청, 정책 제안 (플랫폼 운영)
    # - 일상·공감: 인사, 안부, 축하, 공감, 투자 푸념/감정 토로

    # 투자 관련 키워드 (서비스 카테고리 제외용)
    INVESTMENT_KEYWORDS = [
        "종목", "주가", "매수", "매도", "비중", "수익률", "손절", "익절",
        "차트", "섹터", "ETF", "배당", "PER", "PBR", "코스피", "코스닥",
        "상장", "공시", "실적", "반등", "조정", "상승", "하락", "폭락",
        "투자", "포트폴리오", "비트코인", "가상화폐", "선물", "옵션",
        "전기", "원전", "반도체", "에너지", "바이오", "2차전지",
        "추매", "물타기", "분할매수", "목표가", "손실", "수익",
        # 추가: 수업/강의 내용 관련 투자 용어
        "벨류에이션", "멀티플", "어닝", "회계년도", "선반영",
        "수출", "허가", "칩", "엔비디아", "테슬라", "애플",
        "AI", "알파", "시장", "트레이딩", "매매",
    ]

    TRAINING_EXAMPLES = [
        # ========== 감사·후기 (12개) ==========
        {"content": "감사합니다", "category": "감사·후기"},
        {"content": "정보 감사합니다", "category": "감사·후기"},
        {"content": "답변주셔서 고맙습니다", "category": "감사·후기"},
        {"content": "쌤 덕분에 투자의 눈을 떠가는 1인입니다. 정말 감사합니다.", "category": "감사·후기"},
        {"content": "항상 좋은 정보 감사드립니다. 덕분에 많이 배우고 있습니다.", "category": "감사·후기"},
        {"content": "쌤 믿고 기다리니 쭉쭉 올라갔어요. 신입분들 조급하게 생각마시고 기다리세요.", "category": "감사·후기"},
        {"content": "처음 두산우를 고점에 들어가서 몇달 고생했는데 믿고 기다리니 쭉쭉올라갔어요.", "category": "감사·후기"},
        {"content": "몰빵 종목 털어버리고 새로운 신학기로 가서 좋았어요.", "category": "감사·후기"},
        {"content": "쌤의 엄청난 노력으로 제가 돈 벌고 있음에 고맙고 감사합니다.", "category": "감사·후기"},
        {"content": "월간전망 강의듣고 집으로가는길입니다. 늘 감사드립니다.", "category": "감사·후기"},
        {"content": "오프라인 강의 참석했는데 정말 유익했습니다.", "category": "감사·후기"},
        {"content": "이제는 PB에게 잘 안 물어보고 합니다~~~", "category": "감사·후기"},

        # ========== 질문·토론 (20개) - 투자 질문 + 시장 의견/불안/기대 ==========
        {"content": "삼성전자를 스터디 목록에 편입하지 않으시는 이유가 궁금합니다.", "category": "질문·토론"},
        {"content": "포트폴리오 종목들 비중도 같이 알려주실수있나요?", "category": "질문·토론"},
        {"content": "비트코인 rsi는 일봉 기준인가요?", "category": "질문·토론"},
        {"content": "sk이노베이션은 에너지와 이차전지 혼합인가요?", "category": "질문·토론"},
        {"content": "이수페타시스는 스터디종목에서 제외된걸까요?", "category": "질문·토론"},
        {"content": "지금 현금비중 얼마나 가져가시나요?", "category": "질문·토론"},
        {"content": "수익률의 기준이 되는 종목 매수 매도 단가를 알려주실 수 있을까요?", "category": "질문·토론"},
        {"content": "두산이 반도체섹터인지도 몰랐는데. 반도체섹터 검색하면 안나오더라구요.", "category": "질문·토론"},
        {"content": "트럼프 대통령 뉴스 때문에 주식에 영향이 있을까요? 현금이 없어 걱정되서요.", "category": "질문·토론"},
        {"content": "조정장이 올수도 있다고 하셨는데요 이게 4분기보다 조정이 클까요?", "category": "질문·토론"},
        {"content": "손실이 너무 커서 걱정됩니다. 어떻게 해야할까요?", "category": "질문·토론"},
        {"content": "좋은 종목도 싸게 매수해야 좋은건데 고점에서 손해가 너무 크네요.", "category": "질문·토론"},
        {"content": "ETF 상품이 괴리율을 며칠째 음수로 못 맞추고 있어서 스트레스 받아요.", "category": "질문·토론"},
        {"content": "저는 이미 매수한 상태입니다 ㅠㅠ 어떻게 해야할지 모르겠어요.", "category": "질문·토론"},
        {"content": "크립토는 편출 이슈 이후에 진입해도 늦지 않을까요? 기대됩니다.", "category": "질문·토론"},
        {"content": "원전 월요일 가겠죠? 기다림의 미학입니다.", "category": "질문·토론"},
        {"content": "허싸이 그룹 지금 진입 타이밍 째려보고 있는데 매수 추천구간 말씀해주실 수 있을까요?", "category": "질문·토론"},
        {"content": "계양전기 계속 오릅니다. 뭐든 허투루 보면 안되겠습니다.", "category": "질문·토론"},
        {"content": "전쟁 한시간만에 끝났네요. 시장 어떻게 될까요?", "category": "질문·토론"},
        # 투자 관련 질문 (서비스 문의와 구분)
        {"content": "전기세 동결로 한국전력 주가가 좀 더 오를까요? 추매 해야할지 고민되네요.", "category": "질문·토론"},
        {"content": "템퍼스 AI 주식 스터디 수업 목록에서 뺀 이유가 있을까요?", "category": "질문·토론"},
        {"content": "퇴직금 DB형 DC형 어떤게 나을지 고민입니다.", "category": "질문·토론"},
        {"content": "베네수엘라 사태 어떻게 보시나요?", "category": "질문·토론"},
        {"content": "K~~님 답변 감사합니다. 섹터를 공부해 보려하는데 처음 투자자들은 어느 섹터가 좋을까요?", "category": "질문·토론"},
        {"content": "바이오 섹터 지금 들어가도 될까요? 고점인 것 같아서요.", "category": "질문·토론"},
        {"content": "원전주 언제쯤 다시 반등할까요?", "category": "질문·토론"},
        {"content": "수업에서 다룬 종목 언제 매도하면 좋을까요?", "category": "질문·토론"},
        # 추가: 수업/강의 내용 관련 질문 (서비스 문의와 구분)
        {"content": "수업듣다보면, 회계년도, 처리에 따라 벨류에이션이 변화 한다고 하시는데, 잘이해가 안갑니다.", "category": "질문·토론"},
        {"content": "기업의 어닝이 같은데, 년도가 바뀌면 왜 멀티풀이 변화할까요?", "category": "질문·토론"},
        {"content": "엔비디아 칩 상무부에서 중국 수출 허가 났는데 왜 선반영 안돼요?", "category": "질문·토론"},
        {"content": "강의에서 말씀하신 내용 중에 이해가 안 가는 부분이 있어요.", "category": "질문·토론"},
        {"content": "수업 내용 중 궁금한 게 있는데요, 이 부분 좀 설명해주실 수 있나요?", "category": "질문·토론"},
        # 추가: 게시판/커뮤니티 관련 질문 (서비스 불편과 구분)
        {"content": "게시판 뒤져봐도 그닥 없는거같은데 왜케 난리났어요?", "category": "질문·토론"},
        {"content": "다들 무슨 얘기하는건지 모르겠어요. 뭔 일이에요?", "category": "질문·토론"},
        # 추가: AI/시장 분석 관련 (서비스 불편과 구분)
        {"content": "AI로 분석하면 어떤 결과가 나오나요? 시장 알파가 줄어들 수도 있을까요?", "category": "질문·토론"},
        {"content": "아무 자료도 넣지 않고 분석시킨건데 꽤나 그럴듯하게 만들어주네요.", "category": "질문·토론"},

        # ========== 정보성 글 (14개) ==========
        {"content": "AI 데이터센터 수요가 폭발적으로 증가하고 있습니다.", "category": "정보성 글"},
        {"content": "포트폴리오에서 반도체 비중을 16%에서 28%까지 올렸습니다.", "category": "정보성 글"},
        {"content": "지난밤 미국 원전주 포함 에너지주가 폭등했습니다.", "category": "정보성 글"},
        {"content": "정리된 종목표 챕터에 있습니다 참고요~", "category": "정보성 글"},
        {"content": "속보 떴네요. 뉴스 링크 공유합니다.", "category": "정보성 글"},
        {"content": "작년 코스피 상승, 삼전·SK하닉 빼면 반토막", "category": "정보성 글"},
        {"content": "오늘 코스피 삼전,닉스 빼면 지수가 마이너스. 하락종목수가 더 많아요.", "category": "정보성 글"},
        {"content": "[지노믹트리] 식약처 승인이 연장되더니 결국 취소됐습니다.", "category": "정보성 글"},
        {"content": "현재 PER 34 저평가다. 보수적으로 PER 36으로 계산해보면 18% 추가 상승 가능합니다.", "category": "정보성 글"},
        {"content": "올해 1년간 결산입니다. 한국자산 +166.8%, 해외자산 +23.9% 수익률입니다.", "category": "정보성 글"},
        {"content": "두산 CCL수출 전분기보다 좋음. 다만 시장 기대처럼 판가가 빨리 개선 안되는중~", "category": "정보성 글"},
        {"content": "코스피 하락중입니다. 조정 구간으로 보입니다.", "category": "정보성 글"},
        {"content": "자료 요청 문의가 많은데 12월 24일자 월간 곽수종 28화에 들어가시면 자료 다운로드 받을 수 있어요.", "category": "정보성 글"},
        {"content": "자습차원에서 공유합니다. 출처: 유튜브채널 소수몽키", "category": "정보성 글"},
        {"content": "회원이 아니어도 글쓰기가 되네요. 저도 이제 알았네요~", "category": "정보성 글"},

        # ========== 서비스 문의 (16개) - 서비스 이용 방법, 멤버십, 결제, 기능 질문 ==========
        {"content": "1등 매니저 따라하기 신청 방법 좀 알려주세요~~", "category": "서비스 문의"},
        {"content": "멤버십 만기된거 같은데 자동 연장 되나요?", "category": "서비스 문의"},
        {"content": "멤버십 연장은 자동으로 되나요 공지가 없어서 궁금해요", "category": "서비스 문의"},
        {"content": "개나리반하고 겨울학기하고 어떻게 차이 있나요?", "category": "서비스 문의"},
        {"content": "유료 강좌가 몇 개 있나요?", "category": "서비스 문의"},
        {"content": "결제 방법을 변경하고 싶은데 어떻게 하나요?", "category": "서비스 문의"},
        {"content": "오프라인 강의 초대는 어떻게 받나요?", "category": "서비스 문의"},
        {"content": "컴퍼런스 일정 안내를 받지 못했습니다. 언제 하는지 알 수 있을까요?", "category": "서비스 문의"},
        {"content": "홍매화반 교재 언제 보내주세요?", "category": "서비스 문의"},
        {"content": "책 도착했는데 1권만 배송됐어요. 나머지 2권은 언제 배송될까요?", "category": "서비스 문의"},
        {"content": "오프라인 모임 장소가 어디인가요?", "category": "서비스 문의"},
        {"content": "세금계산서 발행 가능한가요?", "category": "서비스 문의"},
        # 기능 관련 질문 (제보/건의가 아님)
        {"content": "상품 변경 칸이 없어서요~ 변경이 가능할까요?", "category": "서비스 문의"},
        {"content": "녹음 파일이 안보이던데 시간이 더 걸리는 건가요?", "category": "서비스 문의"},
        {"content": "댓글 기능이 없는 이유가 궁금합니다.", "category": "서비스 문의"},
        {"content": "아님 자동 연장이 되나요?", "category": "서비스 문의"},

        # ========== 서비스 불편 (10개) - 플랫폼/앱 버그, 서비스 운영 불만 ==========
        {"content": "어스 플렛폼에서 검색기능 가끔 사용하는데 정렬기능이 없어서 좀 불편합니다. 현재는 날짜가 뒤죽박죽입니다.", "category": "서비스 불편"},
        {"content": "앱이 자꾸 튕겨요. 너무 답답합니다.", "category": "서비스 불편"},
        {"content": "결제를 했는데 강의가 안 열려요. 정말 불편합니다.", "category": "서비스 불편"},
        {"content": "강의 자료 링크가 연결되지 않습니다.", "category": "서비스 불편"},
        {"content": "구독료 대비 콘텐츠가 너무 적은 것 같아요.", "category": "서비스 불편"},
        {"content": "일주일에 한번도 안 올라오는 컨텐츠에 무엇을 기대할까요...", "category": "서비스 불편"},
        {"content": "매번 같은 질문인데 왜 답변을 안 해주시는 건가요? 소통이 안 되는 느낌입니다.", "category": "서비스 불편"},
        {"content": "문의를 드렸는데 답이 없네요. 실망입니다.", "category": "서비스 불편"},
        {"content": "다른 분들은 초대받으셨는데 저는 왜 안오는지 모르겠어요. 소외감이 느껴집니다.", "category": "서비스 불편"},
        {"content": "자동결제되는 시스템은 좀 시정이 되어야 하지않을까요? 환불도 잘 안해줘요.", "category": "서비스 불편"},

        # ========== 서비스 제보/건의 (6개) - 운영진 대상 사칭 신고, 기능 요청, 정책 제안 ==========
        # 핵심: 운영진에게 "제보/신고/요청/제안"하는 내용 (유저 간 대화 X, 질문 X)
        {"content": "유튜브에 사칭하는 광고가 있어 제보합니다.", "category": "서비스 제보/건의"},
        {"content": "유튜브에 박두환님을 사칭하는 광고가 있어 제보합니다. 사기 당할 수 있을 것 같아 우려됩니다.", "category": "서비스 제보/건의"},
        {"content": "마스터님을 사칭하는 계정이 있어서 운영진께 알려드립니다.", "category": "서비스 제보/건의"},
        {"content": "TV모니터 미러링 기능 추가해 주시기를 바래봅니다.", "category": "서비스 제보/건의"},
        {"content": "운영진께 제안합니다. 멤버십회원만 커뮤니티 글쓰기 가능하도록 조정해야할듯합니다.", "category": "서비스 제보/건의"},
        {"content": "회원이 아닌 사람이 글쓰기가 되니 운영자님 조정 부탁드립니다.", "category": "서비스 제보/건의"},

        # ========== 일상·공감 (22개) - 인사, 안부, 축하 + 투자 푸념 + 유저 간 경고/공유 ==========
        {"content": "새해 복 많이 받으세요!", "category": "일상·공감"},
        {"content": "축하드립니다! 너무 너무 축하드려요.", "category": "일상·공감"},
        {"content": "주말은 잘 쉬시길 바랍니다.", "category": "일상·공감"},
        {"content": "안녕하세요 가입인사드립니다. 동행하게 되어 기쁜마음입니다.", "category": "일상·공감"},
        {"content": "1등 매니져 따라하기 3기 가입인사드립니다.", "category": "일상·공감"},
        {"content": "오늘 아침 브리핑 시원합니다~~ 오늘도 홧팅입니다!", "category": "일상·공감"},
        {"content": "오래 기다린만큼 많이 기대됩니다. 열심히 따라해보려구요.", "category": "일상·공감"},
        {"content": "두환쌤 인기 때문인지 이상한 사람들이 많이 생기네요. 마음이 아픕니다.", "category": "일상·공감"},
        {"content": "요즘 부담감이 심해보여 한마디 남깁니다. 너무 큰 책임감으로 스스로를 옥죄지마세요.", "category": "일상·공감"},
        {"content": "쌤 건강 챙기세요!", "category": "일상·공감"},
        # 투자 관련 감정 토로/푸념 (서비스 불편과 구분)
        {"content": "주가의 하루 하루 변동에 일희일비 하다보니 절망감이 들 때가 있어요.", "category": "일상·공감"},
        {"content": "노타 공시를 잘못 읽어서 물렸어요. 저의 실수네요.", "category": "일상·공감"},
        {"content": "손실이 커서 마음이 힘드네요. 버텨야겠죠.", "category": "일상·공감"},
        {"content": "요즘 장이 안 좋아서 힘드시죠? 다들 화이팅입니다.", "category": "일상·공감"},
        {"content": "고점에 물려서 고생중입니다. 버티는 중이에요.", "category": "일상·공감"},
        {"content": "오늘도 빨간불이네요. 멘탈 잡고 갑시다.", "category": "일상·공감"},
        {"content": "시장이 너무 안 좋네요. 힘내세요 여러분.", "category": "일상·공감"},
        {"content": "ETF 괴리율 때문에 스트레스 받는 중입니다. 다들 어떠세요?", "category": "일상·공감"},
        # 유저 간 경고/공유 (운영진 대상 제보 X)
        {"content": "유튜브에 사칭 광고가 많으니 학우분들 조심하세요.", "category": "일상·공감"},
        {"content": "밴드에 사기가 있으니 들어가면 안돼요. 여러분 주의하세요.", "category": "일상·공감"},
        {"content": "블로그에서 좋은 글을 읽고 생각이 많아졌습니다. 공유합니다.", "category": "일상·공감"},
        {"content": "유튜브상에 악성 유투버들이 많네요. 학우분들 현혹되지 마세요.", "category": "일상·공감"},
        # 추가: 영상/모임 후기 (서비스 문의와 구분)
        {"content": "어제 번개전우회 무편집 영상 보고 쥐구멍 찾느랴 혼쭐났어요. 무편집을 생각 못하고, 내 목소리를 듣는게 이리 괴로울 줄...", "category": "일상·공감"},
        {"content": "어제 올라온 영상 봤는데 재밌네요. 좋은 질문의 중요성을 다시 깨달았어요.", "category": "일상·공감"},
        {"content": "1기의 초심으로 돌아가 보렵니다. 올해의 활기찬 시작, 첫 월요일 좋은 하루 되세요!", "category": "일상·공감"},
        {"content": "지난 모임 영상 다시 봤는데 좋았습니다.", "category": "일상·공감"},
        # 추가: 테스트/의미없는 글 (서비스 문의와 구분)
        {"content": "ㅎㅎ 테스트", "category": "일상·공감"},
        {"content": "테스트입니다", "category": "일상·공감"},
        {"content": "ㅎㅎㅎ", "category": "일상·공감"},
        {"content": "ㅋㅋㅋ", "category": "일상·공감"},
        {"content": "...", "category": "일상·공감"},
    ]

    # 감정 라벨 키워드 (질문·토론 카테고리용)
    NEGATIVE_KEYWORDS = [
        "걱정", "불안", "손실", "손해", "스트레스", "ㅠㅠ", "ㅜㅜ", "힘들", "어렵",
        "무섭", "두렵", "망", "폭락", "급락", "하락", "빠지", "떨어",
    ]
    POSITIVE_KEYWORDS = [
        "기대", "희망", "올라", "상승", "오를", "좋을", "기다림", "설레", "신나",
        "화이팅", "홧팅", "가즈아", "갑시다",
    ]

    # 서비스 관련 카테고리 (투자 키워드가 있으면 이 카테고리로 분류 안 함)
    SERVICE_CATEGORIES = ["서비스 문의", "서비스 불편", "서비스 제보/건의"]

    def __init__(self, collection_name: str = "classification_guide_v6", use_llm_fallback: bool = False, confidence_threshold: float = 0.3):
        """
        VectorContentClassifier 초기화

        Args:
            collection_name: ChromaDB 컬렉션 이름
            use_llm_fallback: confidence가 낮을 때 LLM으로 재분류 여부
            confidence_threshold: LLM fallback을 위한 confidence 임계값
        """
        self.collection_name = collection_name
        self.use_llm_fallback = use_llm_fallback
        self.confidence_threshold = confidence_threshold
        self.llm_classifier = None
        self.llm_fallback_count = 0

        self.store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )

        # LLM fallback 활성화시 분류기 초기화
        if use_llm_fallback:
            try:
                self.llm_classifier = ContentClassifier()
            except Exception as e:
                print(f"⚠️ LLM 분류기 초기화 실패: {e}")
                self.use_llm_fallback = False

        # 학습 예제가 저장되어 있는지 확인
        if self.store.collection.count() == 0:
            print("분류 가이드 데이터 초기화 중...")
            self._initialize_training_data()
            print(f"✓ {len(self.TRAINING_EXAMPLES)}개 예제 저장 완료")

    def _initialize_training_data(self):
        """Few-shot 학습 예제를 벡터 스토어에 저장"""
        for i, example in enumerate(self.TRAINING_EXAMPLES):
            self.store.add_content(
                content_id=f"example_{i}",
                text=example["content"],
                metadata={"category": example["category"]}
            )

    def _classify_sentiment(self, content: str) -> str:
        """
        질문·토론 카테고리의 감정 라벨 분류

        Args:
            content: 콘텐츠 텍스트

        Returns:
            "긍정", "부정", "중립" 중 하나
        """
        content_lower = content.lower()

        # 부정 키워드 체크
        negative_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in content_lower)
        # 긍정 키워드 체크
        positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in content_lower)

        if negative_count > positive_count:
            return "부정"
        elif positive_count > negative_count:
            return "긍정"
        else:
            return "중립"

    def _has_investment_keywords(self, content: str) -> bool:
        """
        투자 관련 키워드가 포함되어 있는지 확인

        Args:
            content: 콘텐츠 텍스트

        Returns:
            투자 키워드 포함 여부
        """
        return any(kw in content for kw in self.INVESTMENT_KEYWORDS)

    def _redirect_service_category(self, content: str, original_category: str) -> str:
        """
        투자 키워드가 포함된 경우 서비스 카테고리에서 적절한 카테고리로 재분류

        Args:
            content: 콘텐츠 텍스트
            original_category: 벡터 검색 결과 카테고리

        Returns:
            재분류된 카테고리
        """
        # 질문형 키워드가 있으면 질문·토론
        question_keywords = ["까요", "을까요", "할까요", "나요", "일까요", "인가요", "?"]
        if any(kw in content for kw in question_keywords):
            return "질문·토론"

        # 감정 표현이 있으면 일상·공감
        emotion_keywords = ["힘들", "힘드", "절망", "스트레스", "푸념", "물렸", "고생", "ㅠ", "ㅜ"]
        if any(kw in content for kw in emotion_keywords):
            return "일상·공감"

        # 기본적으로 질문·토론으로 분류
        return "질문·토론"

    def _should_redirect_service_inquiry(self, content: str) -> str:
        """
        서비스 문의로 분류된 경우 재분류가 필요한지 확인

        Args:
            content: 콘텐츠 텍스트

        Returns:
            재분류할 카테고리 (None이면 유지)
        """
        # 테스트성/의미없는 글 → 일상·공감
        test_keywords = ["테스트", "ㅎㅎㅎ", "ㅋㅋㅋ", "...", "ㅎㅎ"]
        if any(kw in content for kw in test_keywords) or len(content.strip()) < 10:
            return "일상·공감"

        # 수업/강의 내용 질문 + 투자 용어 → 질문·토론
        lesson_keywords = ["수업", "강의", "말씀하신", "설명", "이해가"]
        investment_in_content = self._has_investment_keywords(content)
        if any(kw in content for kw in lesson_keywords) and investment_in_content:
            return "질문·토론"

        # 영상/모임 후기 → 일상·공감
        review_keywords = ["영상 보고", "영상 봤", "모임 영상", "무편집", "목소리", "다시 봤"]
        if any(kw in content for kw in review_keywords):
            return "일상·공감"

        # 시장/종목 관련 질문 → 질문·토론
        market_question_keywords = ["선반영", "왜요", "어떻게 되", "왜 안"]
        if any(kw in content for kw in market_question_keywords) and investment_in_content:
            return "질문·토론"

        # 실제 서비스 문의 키워드가 있으면 유지
        real_inquiry_keywords = ["멤버십", "결제", "가입", "환불", "구독", "상품", "배송", "교재", "연장", "방법"]
        if any(kw in content for kw in real_inquiry_keywords):
            return None  # 서비스 문의 유지

        return None  # 기본적으로 유지

    def _should_redirect_service_complaint(self, content: str) -> str:
        """
        서비스 불편으로 분류된 경우 재분류가 필요한지 확인

        Args:
            content: 콘텐츠 텍스트

        Returns:
            재분류할 카테고리 (None이면 유지)
        """
        # 게시판/커뮤니티 관련 질문 (불편 신고 아님) → 질문·토론
        community_question = ["뒤져봐도", "난리", "무슨 일", "뭔 일", "왜케", "왜 그래"]
        if any(kw in content for kw in community_question):
            return "질문·토론"

        # AI/시장 분석 관련 → 질문·토론 또는 정보성 글
        ai_market_keywords = ["AI", "분석", "알파", "시장", "프롬프트", "어드바이저"]
        if any(kw in content for kw in ai_market_keywords) and not self._is_real_complaint(content):
            return "질문·토론"

        # 실제 서비스 불편 키워드가 있으면 유지
        if self._is_real_complaint(content):
            return None  # 서비스 불편 유지

        return None  # 기본적으로 유지

    def _is_real_complaint(self, content: str) -> bool:
        """실제 서비스 불편/불만 키워드가 있는지 확인"""
        real_complaint_keywords = [
            "앱이", "튕겨", "오류", "버그", "안 열려", "안돼", "불편",
            "답답", "실망", "안 해줘", "소외감", "환불", "시정", "결제"
        ]
        return any(kw in content for kw in real_complaint_keywords)

    def _should_redirect_suggestion(self, content: str) -> str:
        """
        서비스 제보/건의로 분류된 경우 재분류가 필요한지 확인

        Args:
            content: 콘텐츠 텍스트

        Returns:
            재분류할 카테고리 (None이면 유지)
        """
        # 유저 간 대화 키워드 → 일상·공감
        user_to_user_keywords = ["학우분들", "여러분", "분들", "조심하세요", "주의하세요", "현혹되지"]
        if any(kw in content for kw in user_to_user_keywords):
            return "일상·공감"

        # 질문 형태 → 서비스 문의
        question_keywords = ["되나요", "인가요", "할까요", "일까요", "궁금합니다", "궁금해요", "가능할까요", "걸리나요"]
        if any(kw in content for kw in question_keywords):
            return "서비스 문의"

        # 단순 정보 공유 → 정보성 글
        info_keywords = ["되네요", "있네요", "알았네요", "있더라구요", "있더라고요"]
        if any(kw in content for kw in info_keywords):
            return "정보성 글"

        # 운영진 대상 제보/요청 키워드가 있으면 유지
        operator_keywords = ["제보합니다", "신고합니다", "운영자님", "운영진께", "추가해 주", "부탁드립니다", "바랍니다"]
        if any(kw in content for kw in operator_keywords):
            return None  # 서비스 제보/건의 유지

        # 기본적으로 일상·공감으로 재분류 (애매한 경우)
        return "일상·공감"

    def classify_content(self, content: str) -> Dict[str, Any]:
        """
        단일 콘텐츠를 벡터 유사도로 분류 (하이브리드: low confidence시 LLM fallback)

        Args:
            content: 분류할 콘텐츠 텍스트

        Returns:
            {"category": str, "confidence": float, "similar_example": str, "method": str, "sentiment": str (질문·토론만)}
        """
        if not content or len(content.strip()) == 0:
            return {
                "category": "내용 없음",
                "confidence": 0.0,
                "similar_example": "",
                "method": "empty",
                "sentiment": None
            }

        # 가장 유사한 예제 검색
        similar = self.store.search_similar(
            query_text=content[:500],  # 처음 500자만 사용
            n_results=1
        )

        if similar and len(similar) > 0:
            top_match = similar[0]
            category = top_match["metadata"].get("category", "미분류")

            # 거리를 confidence로 변환 (거리가 가까울수록 높은 confidence)
            distance = top_match.get("distance", 1.0)
            confidence = max(0.0, min(1.0, 1.0 - distance))

            # LLM fallback: confidence가 낮으면 LLM으로 재분류
            if self.use_llm_fallback and self.llm_classifier and confidence < self.confidence_threshold:
                try:
                    llm_result = self.llm_classifier.classify_content(content)
                    self.llm_fallback_count += 1
                    category = llm_result.get("category", category)
                    confidence = 0.8 if llm_result.get("confidence") == "높음" else 0.6

                except Exception:
                    pass  # LLM 실패시 벡터 결과 사용

            # 키워드 필터링: 서비스 카테고리인데 투자 키워드가 있으면 재분류
            method = "vector"
            if category in self.SERVICE_CATEGORIES and self._has_investment_keywords(content):
                category = self._redirect_service_category(content, category)
                method = "vector+keyword_filter"

            # 서비스 제보/건의 추가 필터링: 유저 간 대화/질문 형태면 재분류
            if category == "서비스 제보/건의":
                redirect_to = self._should_redirect_suggestion(content)
                if redirect_to:
                    category = redirect_to
                    method = "vector+suggestion_filter"

            # 서비스 문의 추가 필터링: 수업 질문, 영상 후기, 테스트 글 등
            if category == "서비스 문의":
                redirect_to = self._should_redirect_service_inquiry(content)
                if redirect_to:
                    category = redirect_to
                    method = "vector+inquiry_filter"

            # 서비스 불편 추가 필터링: 게시판 질문, AI/시장 분석 등
            if category == "서비스 불편":
                redirect_to = self._should_redirect_service_complaint(content)
                if redirect_to:
                    category = redirect_to
                    method = "vector+complaint_filter"

            # 질문·토론 카테고리인 경우 감정 라벨 추가
            sentiment = None
            if category == "질문·토론":
                sentiment = self._classify_sentiment(content)

            return {
                "category": category,
                "confidence": confidence,
                "similar_example": top_match["text"][:100],
                "method": method,
                "sentiment": sentiment
            }
        else:
            return {
                "category": "미분류",
                "confidence": 0.0,
                "similar_example": "",
                "method": "none",
                "sentiment": None
            }

    def classify_batch(
        self,
        contents: List[Dict[str, Any]],
        content_field: str = "message"
    ) -> List[Dict[str, Any]]:
        """
        여러 콘텐츠를 일괄 분류 (하이브리드: 벡터 + LLM fallback)

        Args:
            contents: 분류할 콘텐츠 리스트
            content_field: 콘텐츠 텍스트가 포함된 필드명

        Returns:
            분류 결과가 추가된 콘텐츠 리스트
        """
        results = []
        self.llm_fallback_count = 0  # 카운터 리셋

        for i, item in enumerate(contents):
            content_text = item.get(content_field, "")

            classification = self.classify_content(content_text)

            # 원본 데이터에 분류 결과 추가
            result = item.copy()
            result["classification"] = classification
            results.append(result)

            # 진행 상황 출력 (100건마다)
            if (i + 1) % 100 == 0:
                llm_info = f" (LLM: {self.llm_fallback_count}건)" if self.use_llm_fallback else ""
                print(f"  진행: {i + 1}/{len(contents)} 완료{llm_info}")

        # 최종 LLM 사용 통계
        if self.use_llm_fallback and self.llm_fallback_count > 0:
            print(f"  → LLM fallback 사용: {self.llm_fallback_count}건")

        return results


class ServiceCategoryReviewer:
    """서비스 카테고리(서비스 문의, 서비스 불편) LLM 후처리 검토기"""

    # 전체 카테고리 정의
    CATEGORIES = {
        "감사·후기": "마스터에 대한 감사, 긍정적 피드백, 투자 성과 후기",
        "질문·토론": "포트폴리오, 종목, 투자 전략, 비중, 시장 전망에 대한 질문 및 토론",
        "정보성 글": "투자 경험 공유, 종목 분석, 뉴스/정보 공유, 수익률 공유",
        "서비스 문의": "플랫폼/서비스 기능 문의, 멤버십/결제/배송/일정 관련 문의",
        "서비스 불편": "플랫폼 버그, 서비스 운영 불만, 답변 지연 불만",
        "일상·공감": "새해인사, 안부, 축하, 가입인사, 일상 이야기, 공감 표현, 학우들에게 전하는 말"
    }

    def __init__(self):
        """ServiceCategoryReviewer 초기화"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        self.client = Anthropic(api_key=api_key)

    def review_single(self, content: str, current_category: str) -> Dict[str, Any]:
        """
        단일 콘텐츠의 서비스 카테고리 분류가 맞는지 LLM으로 검토

        Args:
            content: 콘텐츠 텍스트
            current_category: 현재 분류된 카테고리 (서비스 문의 또는 서비스 불편)

        Returns:
            {"correct": bool, "suggested_category": str, "reason": str}
        """
        category_desc = "\n".join([
            f"- {cat}: {desc}"
            for cat, desc in self.CATEGORIES.items()
        ])

        prompt = f"""다음은 금융 콘텐츠 크리에이터 플랫폼의 사용자가 작성한 글입니다.
현재 이 글은 "{current_category}"로 분류되어 있습니다.

[분류 카테고리]
{category_desc}

[분류 기준]
- "서비스 문의": 플랫폼 운영에 관한 실제 문의 (멤버십, 결제, 배송, 오프라인 일정, 기능 사용법 등)
- "서비스 불편": 플랫폼/서비스에 대한 실제 불만 (앱 버그, 콘텐츠 부족, 답변 지연 등)
- 투자/종목/시장에 관한 질문은 "질문·토론"
- 학우들에게 정보를 알려주거나 안부/공감을 나누는 글은 "일상·공감"
- 뉴스나 정보를 공유하는 글은 "정보성 글"

[분류할 내용]
{content[:500]}

위 글이 "{current_category}"로 분류된 것이 맞는지 검토해주세요.
다음 형식으로 답변해주세요:

정확함: [예/아니오]
올바른 카테고리: [카테고리명]
이유: [1문장으로 설명]"""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=200,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = message.content[0].text.strip()
            lines = response_text.split('\n')

            correct = True
            suggested_category = current_category
            reason = ""

            for line in lines:
                if line.startswith('정확함:'):
                    value = line.replace('정확함:', '').strip()
                    correct = value == "예"
                elif line.startswith('올바른 카테고리:'):
                    suggested_category = line.replace('올바른 카테고리:', '').strip()
                elif line.startswith('이유:'):
                    reason = line.replace('이유:', '').strip()

            return {
                "correct": correct,
                "suggested_category": suggested_category,
                "reason": reason
            }

        except Exception as e:
            return {
                "correct": True,  # 오류 시 기존 분류 유지
                "suggested_category": current_category,
                "reason": f"검토 오류: {str(e)}"
            }

    def review_batch(
        self,
        items: List[Dict[str, Any]],
        content_field: str = "message"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        서비스 문의/불편으로 분류된 항목들을 LLM으로 검토하고 필요시 재분류

        Args:
            items: 분류된 콘텐츠 리스트 (classification 필드 포함)
            content_field: 콘텐츠 텍스트 필드명

        Returns:
            (수정된 전체 리스트, 변경된 항목 리스트)
        """
        service_categories = ["서비스 문의", "서비스 불편"]
        changes = []
        reviewed_count = 0

        for item in items:
            classification = item.get("classification", {})
            current_category = classification.get("category", "")

            if current_category not in service_categories:
                continue

            content = item.get(content_field, "") or ""
            if not content:
                continue

            reviewed_count += 1
            review_result = self.review_single(content, current_category)

            if not review_result["correct"]:
                # 카테고리 변경
                old_category = current_category
                new_category = review_result["suggested_category"]

                item["classification"]["category"] = new_category
                item["classification"]["review_changed"] = True
                item["classification"]["original_category"] = old_category
                item["classification"]["review_reason"] = review_result["reason"]

                changes.append({
                    "content": content[:100],
                    "from": old_category,
                    "to": new_category,
                    "reason": review_result["reason"]
                })

            # 진행 상황 출력 (10건마다)
            if reviewed_count % 10 == 0:
                print(f"  검토 진행: {reviewed_count}건 완료 (변경: {len(changes)}건)")

        return items, changes
