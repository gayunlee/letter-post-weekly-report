"""v3c Golden Set 벤치마크 — 우선순위 기반 5분류"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

MODEL_DIR = 'models/v3/topic_v3c/final_model'
V3C_TOPICS = ['운영 피드백', '서비스 피드백', '콘텐츠·투자', '일상·감사', '기타']

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(f'{MODEL_DIR}/category_config.json') as f:
    config = json.load(f)
id_to_cat = {int(k): v for k, v in config['id_to_category'].items()}

# Load golden set
with open('data/gold_dataset/v5_golden_set.json') as f:
    golden = json.load(f)

print('=' * 60)
print('  v3c Golden Set 벤치마크 (우선순위 5분류)')
print('=' * 60)
print(f'\n  Golden set: {len(golden)}건')
print(f'  모델: {MODEL_DIR}')

# Golden set v3c 분포
true_dist = Counter(item['v3c_topic'] for item in golden)
print(f'\n  Golden v3c 정답 분포:')
for t in V3C_TOPICS:
    print(f'    {t}: {true_dist.get(t, 0)}건')

# Predict
y_true = []
y_pred = []
misclassified = []

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
        })

acc = accuracy_score(y_true, y_pred)
correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
report = classification_report(y_true, y_pred, labels=V3C_TOPICS, digits=4, zero_division=0)

print(f'\n  정확도: {acc*100:.1f}% ({correct}/{len(golden)})')
print(f'\n{report}')

# Confusion matrix
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

# Misclassified
print(f'\n  오분류: {len(misclassified)}건')
for m in misclassified:
    print(f"    [{m['true']} -> {m['pred']}] conf={m['conf']:.2f} margin={m['margin']:.2f} | {m['text'][:60]}")

# Comparison
print(f'\n{"="*60}')
print(f'  v2 Topic 정확도:        69.3%')
print(f'  v3 4분류 Golden:        83.3% (클린)')
print(f'  v3 4분류+LLM보정:       85.3%')
print(f'  v3b 5분류 Golden:       80.4%')
print(f'  **v3c 5분류 Golden:     {acc*100:.1f}%**')
print(f'  v2 대비: {(acc-0.693)*100:+.1f}%p')
print(f'  4분류+보정 대비: {(acc-0.853)*100:+.1f}%p')
print(f'  v3b 대비: {(acc-0.804)*100:+.1f}%p')
print(f'{"="*60}')

# Save
os.makedirs('models/v3/topic_v3c', exist_ok=True)
with open('models/v3/topic_v3c/golden_benchmark.json', 'w', encoding='utf-8') as f:
    json.dump({
        'golden_set_size': len(golden),
        'accuracy': acc,
        'correct': correct,
        'v3c_topics': V3C_TOPICS,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'misclassified': misclassified,
        'comparison': {
            'v2': 0.693,
            'v3_4cat_clean': 0.833,
            'v3_4cat_corrected': 0.853,
            'v3b_5cat': 0.804,
            'v3c_5cat': acc,
        }
    }, f, ensure_ascii=False, indent=2)
print(f'\n  저장: models/v3/topic_v3c/golden_benchmark.json')
