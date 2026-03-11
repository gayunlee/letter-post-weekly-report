#!/usr/bin/env python3
"""
Classify chunk_1.json items into 5-category v3b taxonomy.
Uses contextual understanding with manual overrides for edge cases.

Categories:
1. 콘텐츠·투자 - Substantive content about investment/market/stocks/content feedback with details
2. 일상·감사 - Greetings, thanks, encouragement, daily life, self-intro WITHOUT substance
3. 운영 피드백 - Operations issues: seminar inquiries, refund, membership, shipping, pricing
4. 서비스 피드백 - Technical/dev issues: app bugs, payment errors, login, feature requests
5. 기타 - Meaningless noise: ".", "1", consonants only, test posts, under 5 chars no meaning
"""

import json
import re

INPUT_PATH = "/Users/gygygygy/Documents/ai/letter-post-weekly-report/data/training_data/v3/relabel_chunks/chunk_1.json"
OUTPUT_PATH = "/Users/gygygygy/Documents/ai/letter-post-weekly-report/data/training_data/v3/relabel_chunks/chunk_1_classified.json"

# Manual overrides for items that heuristics get wrong
# These are determined by reading the actual text content
MANUAL_OVERRIDES = {
    # Items originally classified as 기타 that should be 콘텐츠·투자
    1009: "콘텐츠·투자",  # "엘지디플도 말씀좀 해주세요" - asking about specific stock
    1041: "콘텐츠·투자",  # "캐시우드님 워렌버핏 명언이네요" - investment context reference
    1051: "콘텐츠·투자",  # Warren Buffett quote about investing
    1074: "콘텐츠·투자",  # "완전 조졌다 ㅠㅠ" - investment distress
    1085: "콘텐츠·투자",  # "은 상승~" - silver price movement
    1087: "콘텐츠·투자",  # "저도 저점에 재진입했네요" - re-entry at low point
    1094: "콘텐츠·투자",  # News about 자사주 소각 의무화 법안
    1107: "콘텐츠·투자",  # "코인 끝인가요" - asking about crypto
    1117: "콘텐츠·투자",  # "ETU는 청산이있을까요?" - investment question
    1181: "콘텐츠·투자",  # "너모 힘들다... 시드는 크지 않다" - investment distress
    1195: "콘텐츠·투자",  # "앞으로 나아가야 할 때" - market context
    1197: "콘텐츠·투자",  # "한번더 속아줘야 하나요?" - investment trust context
    1212: "콘텐츠·투자",  # "인생이 후퇴한 기분" - investment loss distress
    1227: "콘텐츠·투자",  # "차한대 뽑자 가즈아" - investment gains aspiration
    1251: "콘텐츠·투자",  # About 빗썸 비트코인 보유량
    1272: "콘텐츠·투자",  # Criticism of master's investment advice
    1281: "콘텐츠·투자",  # "v자뜨면 바닥 안다지고 찍고 가는거죠?" - market analysis
    1294: "콘텐츠·투자",  # "골드/실버는 목요일 거래량..." - gold/silver analysis
    1301: "운영 피드백",  # "12월호 리미티드 M 안 내놓네" - content delivery complaint
    1308: "콘텐츠·투자",  # "돈많이벌어서 저점일때 삽시다" - investment context
    1314: "운영 피드백",  # "저번 주에도 주말에 나오지 않음?" - content schedule inquiry
    1336: "콘텐츠·투자",  # "장기 빗각인데 하프선 맞고 올라가네요" - chart analysis
    1348: "콘텐츠·투자",  # "결국 미과장님 말대로 된다고!!!" - content + investment
    1352: "콘텐츠·투자",  # "단타나 쳐라" - investment context
    1355: "콘텐츠·투자",  # "오늘은 편안히 잘 수 있겠다" - post-market relief
    1357: "콘텐츠·투자",  # "TLT 태우려고 했었죠?" - bond investment
    1361: "콘텐츠·투자",  # "아이렌이랑 잡주 단타로 -13에서 -7까지" - trading
    1364: "콘텐츠·투자",  # "슬슬 정리를 해볼까...?" - selling consideration
    1366: "콘텐츠·투자",  # "미과장 그렇게 나쁜 사람은 아닐지도?" - master evaluation in investment context
    1370: "콘텐츠·투자",  # "$50k~60k" - bitcoin price target
    1376: "콘텐츠·투자",  # "이악물고 한주도 안팔았음" - holding through downturn
    1384: "콘텐츠·투자",  # "물론 하락은 이제 시작일지 아닐지는" - market outlook
    1386: "콘텐츠·투자",  # "하락 70%, 횡보 30%" - market prediction
    1424: "콘텐츠·투자",  # "-35% 말이 안된다" - portfolio loss
    1440: "콘텐츠·투자",  # "BIL TLT 너뿐이다" - bond investment
    1478: "콘텐츠·투자",  # "200선 내려가면 걍 팔겁니다" - selling plan
    1486: "콘텐츠·투자",  # "전 재산 다 태웠다고" - investment distress
    1491: "콘텐츠·투자",  # "'하방은 닫혔고' 검색" - quoting investment advice
    1498: "콘텐츠·투자",  # "빠질때 걍 훅 빠졌으면" - market sentiment
    1513: "콘텐츠·투자",  # About subscription fee perspective in investment context
    1533: "콘텐츠·투자",  # "몇달 잘참고 기다렸네요. 조금씩 모아갑니다" - accumulation
    1559: "콘텐츠·투자",  # "물리신 분들도 원금회복" - investment recovery
    1565: "콘텐츠·투자",  # Community criticism about investment
    1568: "콘텐츠·투자",  # "미과장 이란 사태 별 거 아닙니다" - quoting investment view
    1576: "콘텐츠·투자",  # "지금 마 15퍼 전후고" - portfolio loss level
    1589: "콘텐츠·투자",  # Wanting to re-read investment analysis
    1595: "운영 피드백",  # "긴급 상황이니 긴급으로 라이브 진행 부탁" - requesting live broadcast
    1610: "콘텐츠·투자",  # "Worst 10 days / best 10 days" - market analysis
    1634: "운영 피드백",  # "게시글...댓글...자료들 삭제하기 바쁘네" - content management complaint
    1644: "콘텐츠·투자",  # "원금 복구……" - investment loss context
    1655: "콘텐츠·투자",  # "조금 전까지는 5까지 갔어여" - market price movement
    1657: "기타",  # Meaningless trolling
    1675: "콘텐츠·투자",  # "지금 미친놈처럼 녹는다" - portfolio melting
    1689: "콘텐츠·투자",  # Reference to scammer in investment context
    1710: "콘텐츠·투자",  # "잘 들으시라고요" - about investment content
    1712: "콘텐츠·투자",  # "도망만 가지마십쇼" - directed at master about investment responsibility
    1715: "콘텐츠·투자",  # "빨리 탈출해서" - about leaving investment position
    1716: "콘텐츠·투자",  # About YouTuber moving money - investment context
    1732: "콘텐츠·투자",  # Recommending another YouTuber for investment analysis
    1765: "콘텐츠·투자",  # "두다리뻗고 밤에 잠은 주무시는지" - about master's responsibility
    1778: "콘텐츠·투자",  # "아프다..두면 오르겟지만 버틸 자신이 없네요"
    1791: "콘텐츠·투자",  # "주봉 기준 하락이 이제 시작임" - chart analysis
    1802: "콘텐츠·투자",  # "기회가 왔다" - investment opportunity
    1805: "콘텐츠·투자",  # "전체자산을 투자했는데 맨탈이라도" - investment distress
    1807: "콘텐츠·투자",  # About master recruiting new members with money implications
    1810: "콘텐츠·투자",  # "너는 어쩔거냐 이제?" - investment context
    1814: "콘텐츠·투자",  # "이걸 내가 당하다니!" - investment loss context
    1818: "콘텐츠·투자",  # "사람 인생 망치는게 이렇게 쉽구나" - investment ruin
    1820: "콘텐츠·투자",  # "이익 내는거 불가고 원금이나 회수" - investment loss
    1823: "콘텐츠·투자",  # "오르긴 하죠?" - asking about market recovery
    1825: "콘텐츠·투자",  # "이정도 예측은 아무도 못한다" - about market prediction
    1835: "콘텐츠·투자",  # Starcraft analogy for investment strategy
    1838: "콘텐츠·투자",  # "오래도 했다 시바꺼 311일 개긑이 날려부려쓰" - investment loss
    1843: "콘텐츠·투자",  # "100%올라야 하는거 알지?" - about recovery math
    1857: "콘텐츠·투자",  # "야엘티가 오를때도 다있네요" - TLT price comment
    1862: "콘텐츠·투자",  # "어떻게 해야될지 도저히 모르겠어요" - investment distress
    1875: "콘텐츠·투자",  # "알아봤어야 했는데... 희대의 사기꾼..." - master criticism
    1879: "콘텐츠·투자",  # "아파요 너무 아파야" - investment pain
    1881: "콘텐츠·투자",  # "답없는놈때문에 작년에 벌어놓은거 다 까먹게" - investment loss
    1892: "콘텐츠·투자",  # "내일 런브 제목은 무슨 카드" - content/investment context
    1893: "콘텐츠·투자",  # "최고의 10일이 오더라도 본전은 가능할까요?"
    1910: "콘텐츠·투자",  # "돈이 얼마나 날라갔을까" - investment losses
    1912: "콘텐츠·투자",  # "계속 하락 알림이 오네여" - market decline
    1914: "콘텐츠·투자",  # "이제 코히런트까지" - quantum computing stock
    1922: "콘텐츠·투자",  # "은 진짜 데드켓바운스였네요" - silver market analysis
    1934: "콘텐츠·투자",  # "이런 구간도 최소한 몇 분할은 해야" - investment strategy
    1937: "콘텐츠·투자",  # "못먹어서 진짜 꼴보기 싫었는데" - about missing investment opportunity
    1940: "콘텐츠·투자",  # "내일보다는 오늘 내리면 안전하니까" - market logic
    1942: "콘텐츠·투자",  # "인구신이 맞는날도 있다니" - market humor
    1945: "콘텐츠·투자",  # "미국 시스템이 붕괴되는 시대는 지났습니다" - market view
    1946: "콘텐츠·투자",  # "하...조짓네" - investment frustration
    1948: "콘텐츠·투자",  # "간당간당하다잉? 종가기준이다" - market close
    1952: "콘텐츠·투자",  # "전부 퍼렇네유" - market red/green
    1965: "콘텐츠·투자",  # "맘은 아프지만 속은 후련하네요" - selling relief
    1969: "콘텐츠·투자",  # "죄송하다고하면 다인거야?" - about master's apology
    1971: "콘텐츠·투자",  # "진짜 미신도들은 능지가" - about blind followers
    1979: "콘텐츠·투자",  # "이미 산사람들은 우짜면 좋을까요" - investment advice seeking
    1981: "콘텐츠·투자",  # "형들 못빠져 나가게 계속 말로 현혹하고" - master criticism
    1982: "콘텐츠·투자",  # "스트레티지 비트마인 하… 비트 이더" - crypto
    1983: "콘텐츠·투자",  # "여러분 죄송합니다는 오늘 나온 런브" - about master's broadcast
    1990: "콘텐츠·투자",  # "삶아지고 있는 3기면 개추" - boiling frog analogy for investment
    1992: "콘텐츠·투자",  # "1년2개월전에 314였는데 지금 4.7" - price comparison
    1995: "콘텐츠·투자",  # "빼요?! 말아요?!" - sell or hold question
    1999: "콘텐츠·투자",  # "내 10개월의 여정 안녕~" - leaving investment membership

    # Items that should be 일상·감사
    1018: "일상·감사",  # "캐시우드님 고맙습니다... 학우님들 수고 많으셨습니다" - pure thanks
    1034: "일상·감사",  # "해결되었습니다^^" - simple acknowledgment
    1039: "기타",  # "운동많이된다" - too short and ambiguous
    1047: "일상·감사",  # About visiting a cafe - daily life
    1058: "일상·감사",  # Asking about a cafe location
    1061: "일상·감사",  # Daily life + greetings
    1063: "일상·감사",  # "감사합니다^^~" - pure thanks
    1095: "일상·감사",  # Self-introduction of 70-year-old without investment substance
    1100: "일상·감사",  # "꾸벅" - bow/greeting
    1109: "일상·감사",  # "감사합니다~ 글 읽어보며 마음 다잡는데 도움이 많이 됩니다" - thanks for emotional support
    1149: "일상·감사",  # "미과장님 울지 말아요. 항상 그랬던 것처럼 잘할거에요" - encouragement
    1189: "일상·감사",  # Birthday congratulations
    1190: "일상·감사",  # "많은 도움이 될듯 합니다" - simple appreciation without substance
    1211: "일상·감사",  # "아자아자 화이팅!!!!!" - pure encouragement
    1249: "일상·감사",  # "응원합니다. 소신껏 해주세요" - pure encouragement
    # 1275: moved to 콘텐츠·투자 below (mentions "좋은 투자전략" + "곧 올라갈거에요" = investment opinion)
    1277: "콘텐츠·투자",  # "오늘 투자쿠키 보는데 울컥해하시네요" - about content + investment sentiment
    1353: "일상·감사",  # "마음이 편안하네요" - emotional state
    1398: "일상·감사",  # "자, 나아갑시다!" - encouragement
    1436: "일상·감사",  # "참으로 오랜만에 실시간으로 채팅도 하고... 좋은 주말 되세요" - greeting
    # 1445: moved to 콘텐츠·투자 below (mentions -2천 investment loss)
    1455: "콘텐츠·투자",  # Asking about how to transcribe lectures - content related
    1456: "운영 피드백",  # Asking about recording availability
    1463: "일상·감사",  # "아자아자!! 모두 힘내봅시다" - encouragement
    1476: "일상·감사",  # "떡볶이를 먹어야겠습니다" - daily life
    1520: "일상·감사",  # "늦점이나 먹자.." - daily life
    1600: "일상·감사",  # "정말 수고 많으셨습니다... 함께 나아가주셔서 감사합니다" - thanks
    1642: "콘텐츠·투자",  # "어쩌죠.. 너무 힘듭니다" - investment distress
    1649: "콘텐츠·투자",  # "누구하나 죽어야 이 사태가 알려지고" - investment anger
    1670: "콘텐츠·투자",  # "많이 다급해보이네" - about master being desperate
    1685: "기타",  # "그냥 궁금해서요" - too vague
    1703: "콘텐츠·투자",  # "그래야 더 열심히 할거같은데" - about master's effort in investment
    1707: "콘텐츠·투자",  # "진짜에요..심신미약이에요 이거" - investment distress
    1731: "콘텐츠·투자",  # Swimming analogy for investment risk
    1740: "콘텐츠·투자",  # "그곳에선 행복하시길" - sarcastic farewell to master
    1749: "기타",  # Consonants only
    1756: "콘텐츠·투자",  # "설렌다 이제 대화가 될까?" - about market dialogue
    1761: "콘텐츠·투자",  # "정말로 죄송하다면 진짜 죄송합니다라고" - about master's apology
    1764: "콘텐츠·투자",  # "결국엔 다 잘되나요" - investment outcome question
    # 1776: moved to 콘텐츠·투자 below (dark humor about investment despair)
    1785: "콘텐츠·투자",  # "진짜 어이가 없어서.." - investment frustration
    1789: "콘텐츠·투자",  # "두근" - anticipation in investment context
    1793: "일상·감사",  # About food - daily life
    1798: "콘텐츠·투자",  # "굉장히 고통스럽네요" - investment pain
    1816: "콘텐츠·투자",  # "할말없음.." - investment speechlessness
    1830: "콘텐츠·투자",  # "아아..." - investment distress
    1841: "콘텐츠·투자",  # "얼마나 성장할ㅋ까" - about growth potential
    1851: "콘텐츠·투자",  # "정신이 이상하다" - investment madness
    1884: "콘텐츠·투자",  # "와우~" - reaction to market movement
    1888: "콘텐츠·투자",  # "말로만 미안하다고 하지말고" - about master's responsibility
    1997: "일상·감사",  # "자랑좀 함 ㅋ" - sharing personal achievement

    # Items that should be 운영 피드백
    # Note: 1024 has both ops (payment) + tech (app) issues; classified as 서비스 피드백 below
    1066: "운영 피드백",  # Asking about AI homepage access for interns
    1070: "운영 피드백",  # New member asking where to start
    1120: "운영 피드백",  # New subscriber asking about curriculum
    1144: "운영 피드백",  # Asking about how to watch Monday live on Zoom
    1145: "운영 피드백",  # "게시글을 어떻게 볼수있나요?" - how to use platform
    1155: "운영 피드백",  # Asking about community rules/permissions
    1161: "운영 피드백",  # Preparing welcome materials for new students
    1201: "운영 피드백",  # Asking when live recording will be uploaded
    1264: "운영 피드백",  # New member asking about 말머리 usage
    1298: "운영 피드백",  # Asking about replay availability
    1423: "운영 피드백",  # "시간이 안되어서 참여하질 못했네요" - about attendance
    1532: "운영 피드백",  # "올려주시면 감사하겠습니다" - requesting content upload
    1560: "운영 피드백",  # "네이키드 인싸이트 안하나요" - content schedule inquiry

    # Items that should be 서비스 피드백
    1024: "서비스 피드백",  # Mix of ops + tech: app not opening is more tech
    1071: "서비스 피드백",  # Audio playback bug with detailed reproduction steps

    # Fix: items in 기타 that should be 콘텐츠·투자 (missed in first pass)
    1106: "콘텐츠·투자",  # "로블록스 계속 빠지는데 계속 가져가도 되는건가요?" - stock question
    1721: "콘텐츠·투자",  # "고점대비 에센피 삼프로하락" - S&P market data
    1726: "콘텐츠·투자",  # "난 죽어도 롱에 죽어.. 쫄려도 just keep buying" - investment stance
    1729: "콘텐츠·투자",  # "아이렌 숏 샀는데 50%먹고 이제 거의 원금 회복했네요" - trading result
    1790: "콘텐츠·투자",  # "내가 어제도 말햇고 더 떨어진다고 숏잡으라고 ㅋㅋ" - trading advice
    1829: "콘텐츠·투자",  # "하락의 가속화 미첬네;;" - market commentary
    1889: "콘텐츠·투자",  # "그냥 지금부터 아이렌이나 스트래티지 숏치죠?" - trading question

    # Fix: items in 서비스 피드백 that should be other categories
    1435: "콘텐츠·투자",  # "소담소담 라이브... 유익하고 좋습니다" - positive content feedback
    1720: "콘텐츠·투자",  # "삼전과 한에어가 하한가 거래가 됐는데" - stock market question
    1737: "콘텐츠·투자",  # Community reflection during bad market - investment context
    1794: "콘텐츠·투자",  # "전재산 투자를... 1기 수익률" - investment loss story/criticism

    # Fix: items in 운영 피드백 that should be 콘텐츠·투자
    1343: "콘텐츠·투자",  # "변동성이 적고 크게 상승하는 우량주 나스닥 반도체" - investment advice
    1509: "콘텐츠·투자",  # Commentary about investment responsibility
    1774: "콘텐츠·투자",  # Defending master's investment approach
    1813: "콘텐츠·투자",  # "투자의 책임은 전적으로 본인에게" - investment philosophy
    1876: "콘텐츠·투자",  # Criticism of master's attitude about investment
    1747: "콘텐츠·투자",  # "창립멤버로 참여했습니다... 리스크를 함께 감수하겠다" - investment stance

    # Fix: items in 일상·감사 that should be 콘텐츠·투자
    1062: "콘텐츠·투자",  # "학우님들 변동성 신경쓰지 마시고... 기업들 잘 성장" - has investment substance
    1275: "콘텐츠·투자",  # "충분히 좋은 투자전략... 곧 올라갈거에요" - investment opinion/defense
    1445: "콘텐츠·투자",  # "-2천이나 -2천10만원이나 또이또이다" - mentions investment loss numbers
    1776: "콘텐츠·투자",  # "번개탄이랑 부루스타좀 사게" - dark humor about investment loss despair on 미과장 board
    # Note: 1786 is high school student encouragement - stays as 일상·감사 (no investment substance)

    # Various corrections
    1054: "콘텐츠·투자",  # "오늘 같은 날은 햇살 두환쌤님의 아침 수업 들으며" - about content
    1059: "콘텐츠·투자",  # "소낙비는 일단 피하고 봐야겠네요" - market analogy
    # 1062: moved to 콘텐츠·투자 above (mentions 변동성, 기업 성장 = investment substance)
    1102: "콘텐츠·투자",  # About buying robot at Emart - relates to robotics investment
    1103: "콘텐츠·투자",  # Political/economic news about negotiations
    1104: "콘텐츠·투자",  # Trump Iran geopolitics affecting markets
    1133: "기타",  # About ramen flavor - daily life trivia
    1135: "일상·감사",  # Recommending a printer product - daily life sharing
    1139: "일상·감사",  # Money counting story - parable/encouragement
    1152: "일상·감사",  # New semester welcome preparation - community building
    1157: "기타",  # "그러하다" - meaningless
    1169: "기타",  # "미과장 밖에 안보이고" - too vague without context
    1182: "기타",  # "이라고 합니다. 공부기업들은 아니지만" - fragment
    1198: "기타",  # "주소좀 알려주실분" - too vague
    1204: "콘텐츠·투자",  # About Samsung factory expansion - industry insight
    1209: "기타",  # Just a YouTube link
    1265: "일상·감사",  # "저에게 간절한 소망이 생겼습니다" - emotional sharing without substance
    1278: "콘텐츠·투자",  # "투자쿠키 보니까 또 설득되네" - about investment content
    1287: "일상·감사",  # Asking about robot vacuum recommendations
    1293: "기타",  # Political rant unrelated to investment
    1302: "기타",  # Political discussion about deleted post
    1304: "기타",  # "치고 박고ㅇㅇ" - meaningless
    1306: "일상·감사",  # Weekend cafe visit - daily life with slight inv mention
    1310: "콘텐츠·투자",  # Cooking analogy for investment lessons
    1311: "기타",  # Just a news link with "이건가요?"
    1316: "콘텐츠·투자",  # Snow + Nvidia rising + market
    1331: "기타",  # "와 알바들 역하네;;" - vague community complaint
    1340: "기타",  # "무제... 나름 예의있게 적은 것 같은데" - meta discussion
    1341: "기타",  # "그래서 좀 늦는 듯?" - vague
    1354: "콘텐츠·투자",  # "나포함" - in investment context (me too)
    1358: "기타",  # "찌라시였습니다ㅠ" - false rumor acknowledgment
    1360: "기타",  # About edited/distorted messages
    1372: "콘텐츠·투자",  # "하…" - in investment distress context
    1375: "콘텐츠·투자",  # "그냥 얘기도 꺼내지마" - about investment topic
    1378: "콘텐츠·투자",  # "제뉴노말인것같습니다" - new normal in market context
    1385: "콘텐츠·투자",  # "님들 생각은 어떰?" - asking opinions in investment context
    1392: "기타",  # About YouTube/DC comments being harsh
    1400: "기타",  # "ㄷㄷㄷ" - reaction (meaningless)
    1404: "콘텐츠·투자",  # "내가 병신이지" - investment self-blame
    1406: "콘텐츠·투자",  # "반성문이라도 써오세요" - directed at master
    1407: "기타",  # Political discussion
    1408: "기타",  # "너검마가 아냐?" - unclear
    1425: "기타",  # "다 먹고살자고 하는 일" - vague
    1438: "기타",  # "기다리고있습니다" - too short
    1441: "콘텐츠·투자",  # "진짜 오냐!!!!!" - market excitement
    1447: "콘텐츠·투자",  # "물 떠놓고 기도합시다" - hoping for market
    1450: "콘텐츠·투자",  # Date ranges - investment timeline
    1452: "콘텐츠·투자",  # "개그치 올라온다" - about market rally
    1458: "일상·감사",  # About exercise and stress - daily life
    1492: "콘텐츠·투자",  # "검색하고 들어오세요" - about investment knowledge
    1516: "기타",  # Consonants only
    1517: "콘텐츠·투자",  # "흑흑흑" - crying about investment
    1522: "콘텐츠·투자",  # "하지만 살 돈은 없습니다" - no money to buy dip
    1524: "기타",  # "아... 알겠어 재미나이야..." - vague
    1535: "기타",  # "ㅎㅎ"
    1547: "콘텐츠·투자",  # "죽고싶을거야" - extreme investment distress
    1558: "콘텐츠·투자",  # "게시판 보게 되고 그러네" - checking board for investment
    1592: "기타",  # "그냥 그렇다고요" - meaningless
    1596: "콘텐츠·투자",  # "겁쟁이랍니다아~~~" - about being fearful in market
    1598: "콘텐츠·투자",  # "??? : 제가 이길 수 있습니다" - market bravado
    1611: "콘텐츠·투자",  # "이게 다에요..?!" - about insufficient content/returns
    1627: "기타",  # "ㅇ ㅇ" - meaningless
    1638: "콘텐츠·투자",  # About people screenshotting posts - master accountability
    1962: "기타",  # Curse word
    # 1054: duplicate removed (already listed above)
    1186: "일상·감사",  # Health/rest announcement from community member
    1202: "콘텐츠·투자",  # Sharing baby's story but mixed with investment journey and Samsung shares
    1238: "콘텐츠·투자",  # Reading group discussion post - investment related
    1148: "콘텐츠·투자",  # "50만원 넘게 넣어서 수익은 0" - investment loss complaint about master
    1150: "콘텐츠·투자",  # "트럼프... 이더리움 탑승 좀 더 해볼까" - crypto investment
    1153: "콘텐츠·투자",  # Requesting book review content - investment education
    1154: "콘텐츠·투자",  # Questioning master's credentials and portfolio
    1160: "콘텐츠·투자",  # "하방이 막힌 안전한 투자라며?" - criticism of investment advice
    1177: "콘텐츠·투자",  # "온통 주식이야기…" - market mania observation
    1240: "콘텐츠·투자",  # "같이 이겨냅시다 계좌공개 부탁" - requesting portfolio disclosure
    1241: "콘텐츠·투자",  # "눈가리고 다트던져서 종목맞추는 원숭이" - criticism of stock picking
    1247: "콘텐츠·투자",  # About 대깨미 and emotional followers - investment context
    1256: "콘텐츠·투자",  # Requesting master's account verification
    1258: "콘텐츠·투자",  # "안정적인 투자를 합니다 포트폴리오는 개박살" - investment criticism
    1274: "콘텐츠·투자",  # Long investment philosophy post
    1279: "콘텐츠·투자",  # "-50퍼 찍고 반등하는것. 맞죠?" - market question
    1309: "콘텐츠·투자",  # Criticism of master's behavior and investment advice
    1321: "콘텐츠·투자",  # About selling when 리게티 was up 600%
    1322: "콘텐츠·투자",  # Harsh criticism of master and investment community
    1323: "콘텐츠·투자",  # About 주간보고서 not being uploaded
    1327: "콘텐츠·투자",  # Community complaint about destructive posts + investment distress
    1329: "콘텐츠·투자",  # About account number changes and manipulation
    1330: "콘텐츠·투자",  # "이제는 반등해도 비관적" - investment sentiment
    1234: "콘텐츠·투자",  # About Micron HBM4 supply
    1257: "콘텐츠·투자",  # Defending master, investment context
    1269: "콘텐츠·투자",  # Long rant defending risk-taking in investment
    1276: "콘텐츠·투자",  # "결국 미과장의 인사이트를 믿고" - investment reflection
}

def classify_text(text, master_name, source):
    """Classify a single text into v3b_topic category."""
    cleaned = text.strip()

    # === 기타: meaningless noise ===
    if len(cleaned) <= 2:
        return "기타"
    if re.match(r'^[\.\,\?\!\;\:\-\~\s]+$', cleaned):
        return "기타"
    if re.match(r'^[ㄱ-ㅎㅏ-ㅣ\s\.\,\!\?\~]+$', cleaned) and len(cleaned) <= 10:
        return "기타"
    meaningless_patterns = [r'^\.+$', r'^\?+$', r'^\!+$', r'^[0-9]+$',
                            r'^ㅋ+$', r'^ㅎ+$', r'^ㅠ+$', r'^ㅜ+$']
    for pat in meaningless_patterns:
        if re.match(pat, cleaned):
            return "기타"

    text_lower = cleaned.lower()

    # === 서비스 피드백 ===
    service_keywords = ['버그', '오류', '에러', 'bug', '앱이 안', '앱이 열리지',
        '로그인', '접속이 안', '접속 불가', '프로그램에 버그',
        '화면이 안', '결제 오류', '결제가 안', '사칭', '피싱', '해킹']
    for kw in service_keywords:
        if kw in text_lower:
            return "서비스 피드백"
    if ('재생' in text_lower and ('멈춤' in text_lower or '멈' in text_lower)):
        return "서비스 피드백"

    # === 운영 피드백 ===
    ops_keywords = ['환불', '해지', '탈퇴', '구독취소', '구독 취소',
        '자동결제', '자동결재', '자동이체']
    for kw in ops_keywords:
        if kw in text_lower:
            # Check it's about their own subscription/refund, not market discussion
            if '멤버십 해지' in text_lower or '구독' in text_lower or '결제' in text_lower or '결재' in text_lower:
                return "운영 피드백"
    if ('구독' in text_lower and ('변경' in text_lower or '해지' in text_lower or '취소' in text_lower)):
        return "운영 피드백"
    if ('회원' in text_lower and ('탈퇴' in text_lower)):
        return "운영 피드백"
    if ('회비' in text_lower and ('자동' in text_lower or '결제' in text_lower or '출금' in text_lower)):
        return "운영 피드백"
    if ('업로드' in text_lower and ('언제' in text_lower or '될까' in text_lower)):
        return "운영 피드백"
    if ('다시보기' in text_lower or '녹화본' in text_lower):
        if ('언제' in text_lower or '가능' in text_lower or '할 수' in text_lower or '수강' in text_lower):
            return "운영 피드백"

    # === Investment keywords ===
    investment_keywords = [
        '매수', '매도', '손절', '익절', '추매', '물타기', '물타', '불타기',
        '포트폴리오', '포폴', '포트', '비중', '수익률', '수익율',
        '종목', '주가', '주식', '코스피', '코스닥', '나스닥', 's&p', 'snp', '슨피',
        '배당', '실적', '영업이익', '매출', '순이익', 'per', 'pbr',
        '차트', '이평선', '지지선', '저항선', '캔들', 'rsi',
        '외국인', '기관', '수급',
        '금리', '유동성', '양적완화',
        '환율', '달러', '원화', '엔화',
        '반도체', 'hbm', '메모리', '파운드리', 'dram', 'nand',
        '원전', '원자력', '방산', 'ai', '인공지능', '양자',
        '비트코인', '이더리움', '코인', '크립토', '암호화폐',
        '채권', '국채', 'etf',
        '테슬라', '엔비디아', '삼성전자', '하이닉스', '알파벳', '구글',
        '마이크로소프트', '아마존', '애플', '메타', '팔란티어',
        '두산', '한전', 'sk', 'lg', 'dl', 'hdc',
        '밸류에이션', '내재가치', '저평가',
        '리밸런싱', '분할매수', 'dca',
        '상승장', '하락장', '조정', '폭락', '반등', '급등', '급락',
        '변동성', 'vix', '사이드카',
        '관세', '트럼프', '연준', '파월', '워시',
        '레버리지', '인버스', '곱버스',
        'arkk', 'arkf', 'mags', 'tlt',
        '시총', '선물', '옵션', '블록딜',
        '사이클', '슈퍼사이클',
        '배전', '송전', '데이터센터', 'capex',
        '계좌', '평단', '양전', '음전',
        '마이너스', '수익', '손실',
        '투자쿠키', '주간보고서', '브리핑', '런브',
        'ism', 'mmf', 'm2',
        '오투폴리오', '오투',
        '유가', '금값', '은값',
        '동행학교', '투자학교', '경제학교',
        '빅테크', 'm7',
        '멤버십', '구독료',
        '시장', '장이', '장중', '장세', '미장', '국장',
        '하방', '상방', '눌림목', '지수',
        '공포', '탐욕', '패닉', '패닉셀',
        '전고점', '바닥', '꼭지',
        '분할', '존버', '홀딩', 'hodl',
        '전략',
    ]

    content_keywords = ['강의', '수업', '라이브', '영상', '유튜브',
        '분석', '전망', '인사이트', '뷰', '리포트', '레포트']

    greeting_keywords = ['감사합니다', '고맙습니다', '수고하셨', '수고 많으셨',
        '화이팅', '힘내세요', '응원합니다', '명절', '새해', '설날',
        '건강하세요', '좋은 하루', '좋은 주말', '편안한']

    daily_keywords = ['맛있는', '커피', '카페', '식당', '음식', '날씨', '산책',
        '운동', '조깅', '가족', '아이', '손자', '생일', '축하', '여행']

    inv_count = sum(1 for kw in investment_keywords if kw in text_lower)
    content_count = sum(1 for kw in content_keywords if kw in text_lower)
    greeting_count = sum(1 for kw in greeting_keywords if kw in text_lower)
    daily_count = sum(1 for kw in daily_keywords if kw in text_lower)

    # Very short texts
    if len(cleaned) < 15:
        if inv_count >= 1:
            return "콘텐츠·투자"
        if greeting_count >= 1:
            return "일상·감사"
        return "기타"

    # Strong investment signal
    if inv_count >= 3:
        return "콘텐츠·투자"
    if content_count >= 1 and inv_count >= 1:
        return "콘텐츠·투자"
    if inv_count >= 2:
        return "콘텐츠·투자"
    if inv_count >= 1 and len(cleaned) > 20:
        return "콘텐츠·투자"
    if content_count >= 1 and len(cleaned) > 50:
        # Content with some detail
        detail_kw = ['공부', '배우', '배웠', '도움', '유익', '납득', '이해', '설명', '내용', '수강', '복습']
        if any(kw in text_lower for kw in detail_kw):
            return "콘텐츠·투자"
    if ('http' in text_lower or 'news' in text_lower) and inv_count >= 1:
        return "콘텐츠·투자"

    # Pure greeting/encouragement
    if greeting_count >= 1 and inv_count == 0:
        if daily_count >= 1 or len(cleaned) < 80:
            return "일상·감사"
        if len(cleaned) > 100:
            if any(kw in text_lower for kw in ['계좌', '주가', '수익', '손실', '투자', '주식']):
                return "콘텐츠·투자"
        return "일상·감사"

    if daily_count >= 2 and inv_count == 0:
        return "일상·감사"

    if inv_count >= 1:
        return "콘텐츠·투자"

    # Medium-length text
    if len(cleaned) < 30:
        if greeting_count >= 1:
            return "일상·감사"
        return "기타"

    # Longer texts
    implicit_inv = ['사이클', '유동성', '전고점', '바닥', '꼭지', '분할',
        '버티', '존버', '홀딩', '선반영', '비중', '공부', '학습']
    if any(kw in text_lower for kw in implicit_inv):
        return "콘텐츠·투자"

    if greeting_count >= 1:
        return "일상·감사"

    if len(cleaned) > 50:
        return "콘텐츠·투자"

    return "기타"


def main():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items")

    counts = {"콘텐츠·투자": 0, "일상·감사": 0, "운영 피드백": 0, "서비스 피드백": 0, "기타": 0}

    for item in data:
        idx = item["idx"]
        if idx in MANUAL_OVERRIDES:
            topic = MANUAL_OVERRIDES[idx]
        else:
            topic = classify_text(item["text"], item.get("masterName", ""), item.get("_source", ""))
        item["v3b_topic"] = topic
        counts[topic] += 1

    print(f"\nClassification results:")
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(data)*100:.1f}%)")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nOutput written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
