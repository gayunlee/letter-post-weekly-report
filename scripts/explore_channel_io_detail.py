"""채널톡 상세 탐색: tags + 완전한 대화 샘플"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.bigquery.client import BigQueryClient

client = BigQueryClient()
P = "us-service-data"
D = "channel_io"

# 1. tags 분포
print("=== tags 분포 ===")
for r in client.execute_query(f"""
    SELECT tags, COUNT(*) as cnt
    FROM `{P}.{D}.chats`
    WHERE tags IS NOT NULL AND tags != '[]'
    GROUP BY tags ORDER BY cnt DESC LIMIT 30
"""):
    print(f"  {r['tags']:70s} | {r['cnt']:>4,}건")

print()

# 2. chats.name 분포 (대화 이름/제목)
print("=== chats.name 분포 (상위 20) ===")
for r in client.execute_query(f"""
    SELECT name, COUNT(*) as cnt
    FROM `{P}.{D}.chats`
    WHERE name IS NOT NULL AND name != ''
    GROUP BY name ORDER BY cnt DESC LIMIT 20
"""):
    name = r['name'][:60] if r['name'] else "(null)"
    print(f"  {name:62s} | {r['cnt']:>5,}건")

print()

# 3. 완전한 대화 1건 (closed + tags 있는 것)
print("=== 완전한 문의 건 샘플 ===")
samples = client.execute_query(f"""
    WITH good_chats AS (
        SELECT c.id, c.state, c.tags, c.name,
            TIMESTAMP_MILLIS(c.createdAt) as chat_created,
            c.replyCount
        FROM `{P}.{D}.chats` c
        WHERE c.state = 'closed' AND c.tags IS NOT NULL AND c.tags != '[]'
        ORDER BY c.createdAt DESC
        LIMIT 2
    )
    SELECT
        gc.id as chatId, gc.state, gc.tags, gc.name as chat_name,
        gc.chat_created, gc.replyCount,
        m.personType, m.plainText,
        TIMESTAMP_MILLIS(m.createdAt) as msg_time
    FROM good_chats gc
    JOIN `{P}.{D}.messages` m ON gc.id = m.chatId
    ORDER BY gc.chat_created DESC, m.createdAt ASC
""")

current_chat = None
for r in samples:
    if r['chatId'] != current_chat:
        current_chat = r['chatId']
        print(f"\n--- chatId: {r['chatId']} ---")
        print(f"  state: {r['state']} | tags: {r['tags']}")
        print(f"  name: {r['chat_name']}")
        print(f"  created: {r['chat_created']} | replyCount: {r['replyCount']}")
        print()

    text = (r['plainText'] or '(빈 메시지)')[:200]
    role = '고객' if r['personType'] == 'user' else r['personType']
    print(f"  [{r['msg_time']}] {role}")
    print(f"    {text}")
    print()

# 4. 고객의 실제 문의 내용 (워크플로우 버튼이 아닌 자유 텍스트)
print("=== 고객 자유 텍스트 (버튼 선택 아닌 실제 문의) 샘플 ===")
free_texts = client.execute_query(f"""
    WITH user_msgs AS (
        SELECT chatId, plainText, createdAt,
            ROW_NUMBER() OVER(PARTITION BY chatId ORDER BY createdAt) as msg_order
        FROM `{P}.{D}.messages`
        WHERE personType = 'user'
            AND plainText IS NOT NULL
            AND LENGTH(plainText) > 30
            AND plainText NOT LIKE '%빠른 문의%'
            AND plainText NOT LIKE '%구독신청%'
            AND plainText NOT LIKE '%기타 문의%'
            AND plainText NOT LIKE '%상품변경%'
            AND plainText NOT LIKE '%회원가입%'
            AND plainText NOT LIKE '%이용방법%'
            AND plainText NOT LIKE '%1:1 상담%'
    )
    SELECT plainText, chatId
    FROM user_msgs
    WHERE msg_order <= 2
    ORDER BY createdAt DESC
    LIMIT 15
""")
for r in free_texts:
    text = r['plainText'][:150]
    print(f"  [{r['chatId'][:12]}...] {text}")
    print()
