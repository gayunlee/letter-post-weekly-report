"""v3c Balanced Golden Set 벤치마크 — 균형 학습 모델 + v6 Golden set"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

MODEL_DIR = 'models/v3/topic_v3c_balanced/final_model'
V3C_TOPICS = ['운영 피드백', '서비스 피드백', '콘텐츠·투자', '일상·감사', '기타']

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(f'{MODEL_DIR}/category_config.json') as f:
    config = json.load(f)
id_to_cat = {int(k): v for k, v in config['id_to_category'].items()}

with open('data/gold_dataset/v6_golden_set.json') as f:
    golden = json.load(f)

print('=' * 60)
print('  v3c Balanced Golden Set 벤치마크')
print('=' * 60)
print(f'\n  Golden set: {len(golden)}건')
print(f'  모델: {MODEL_DIR}')

true_dist = Counter(item['v3c_topic'] for item in golden)
print(f'\n  Golden v3c 정답 분포:')
for t in V3C_TOPICS:
    print(f'    {t}: {true_dist.get(t, 0)}건')

y_true, y_pred, misclassified = [], [], []

for item in golden:
    true_topic = item['v3c_topic']
    text = item['text']

    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

    topk = torch.topk(probs, k=2)
    pred_id = topk.indices[0].item()
    pred_conf = topk.values[0].item()
    top2_id = topk.indices[1].item()
    margin = pred_conf - topk.values[1].item()

    pred_topic = id_to_cat[pred_id]
    top2_topic = id_to_cat[top2_id]

    y_true.append(true_topic)
    y_pred.append(pred_topic)

    if true_topic != pred_topic:
        misclassified.append({
            'text': text[:80],
            'true': true_topic,
            'pred': pred_topic,
            'conf': round(pred_conf, 3),
            'margin': round(margin, 3),
            'top2': top2_topic,
            'source': item.get('_golden_source', 'unknown'),
        })

acc = accuracy_score(y_true, y_pred)
correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
report = classification_report(y_true, y_pred, labels=V3C_TOPICS, digits=4, zero_division=0)

print(f'\n  정확도: {acc*100:.1f}% ({correct}/{len(golden)})')
print(f'\n{report}')

cm = confusion_matrix(y_true, y_pred, labels=V3C_TOPICS)
print('  Confusion Matrix:')
header = f"  {'':>14}"
for t in V3C_TOPICS:
    header += f"  {t[:6]:>6}"
print(header)
for i, row in enumerate(cm):
    line = f"  {V3C_TOPICS[i]:>14}"
    for val in row:
        line += f"  {val:>6}"
    print(line)

# Human-verified vs auto-sampled breakdown
human_true, human_pred = [], []
auto_true, auto_pred = [], []
for item, yt, yp in zip(golden, y_true, y_pred):
    if item.get('_golden_source') == 'human_verified':
        human_true.append(yt)
        human_pred.append(yp)
    else:
        auto_true.append(yt)
        auto_pred.append(yp)

if human_true:
    h_acc = accuracy_score(human_true, human_pred)
    print(f'\n  사람 검수 (기존 102건): {h_acc*100:.1f}% ({sum(1 for a,b in zip(human_true,human_pred) if a==b)}/{len(human_true)})')
if auto_true:
    a_acc = accuracy_score(auto_true, auto_pred)
    print(f'  자동 샘플 (신규 125건): {a_acc*100:.1f}% ({sum(1 for a,b in zip(auto_true,auto_pred) if a==b)}/{len(auto_true)})')

print(f'\n  오분류: {len(misclassified)}건')
for m in misclassified:
    src = 'H' if m['source'] == 'human_verified' else 'A'
    print(f"    [{src}] [{m['true']} -> {m['pred']}] conf={m['conf']:.2f} margin={m['margin']:.2f} | {m['text'][:55]}")

print(f'\n{"="*60}')
print(f'  v2 Topic 정확도:        69.3%')
print(f'  v3c 편향 모델 Golden:   85.3% (102건)')
print(f'  **v3c 균형 모델 Golden: {acc*100:.1f}% ({len(golden)}건)**')
print(f'{"="*60}')

os.makedirs('models/v3/topic_v3c_balanced', exist_ok=True)
with open('models/v3/topic_v3c_balanced/golden_benchmark.json', 'w', encoding='utf-8') as f:
    json.dump({
        'golden_set_size': len(golden),
        'accuracy': acc,
        'correct': correct,
        'misclassified': misclassified,
        'human_verified_accuracy': h_acc if human_true else None,
        'auto_sampled_accuracy': a_acc if auto_true else None,
    }, f, ensure_ascii=False, indent=2)
print(f'\n  저장: models/v3/topic_v3c_balanced/golden_benchmark.json')
