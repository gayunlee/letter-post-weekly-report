"""Golden set v3b 재분류 — 콘텐츠·투자 경계 항목을 일상·감사로 재분류"""
import json
from collections import Counter

with open('data/gold_dataset/v5_golden_set.json') as f:
    golden = json.load(f)

# 콘텐츠·투자 → 일상·감사로 재분류할 인덱스
# 기준: 실질적 투자/콘텐츠 내용이 없는 감사/인사/응원/격려
changes = {
    26: '일상·감사',   # '학우분들님 감사합니다^^'
    37: '일상·감사',   # 종교적 기도+건강 기원
    58: '일상·감사',   # '담쌤 오늘 목소리에 기운이 너무 없으세요'
    73: '일상·감사',   # '반성하고 반성했습니다... 공부도 더 열심히하고'
    74: '일상·감사',   # '늘 감명깊게..강의 잘 듣고 있습니다. 감사합니다 ^^'
    77: '일상·감사',   # '동행이라는 단어가 참 좋아요' 응원+감사
    78: '일상·감사',   # '매일 올려주시는 덕분에 도움 받고 있어요' 감사만
    69: '일상·감사',   # '미과장님 뷰대로 흔들리지말고 가주세요' 응원
    60: '일상·감사',   # '용기를 가지고 나아가야합니다' 격려
}

# 운영 피드백 재분류
# [42] '이런 날들은 사원들도 궁금증 듣게 해줘야' → 서비스 운영 방식 제안 → 운영 피드백 유지

changed = 0
for idx, new_topic in changes.items():
    old = golden[idx].get('v3b_topic')
    if old != new_topic:
        golden[idx]['v3b_topic'] = new_topic
        text_short = golden[idx]['text'][:50].replace('\n', ' ')
        print(f'  [{idx}] {old} -> {new_topic} | {text_short}')
        changed += 1

print(f'\n변경: {changed}건')

dist = Counter(item['v3b_topic'] for item in golden)
print(f'\nGolden v3b 최종 분포:')
for t, c in dist.most_common():
    print(f'  {t}: {c}건')

with open('data/gold_dataset/v5_golden_set.json', 'w', encoding='utf-8') as f:
    json.dump(golden, f, ensure_ascii=False, indent=2)
print('저장 완료')
