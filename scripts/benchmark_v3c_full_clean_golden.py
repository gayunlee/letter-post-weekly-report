"""v3c_full_clean Golden Set 벤치마크 — 전수 검수 227건"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from collections import Counter

MODEL_DIR = 'models/v3/topic_v3c_full_clean/final_model'
V3C_TOPICS = ['운영 피드백', '서비스 피드백', '콘텐츠·투자', '일상·감사', '기타']

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(f'{MODEL_DIR}/category_config.json') as f:
    config = json.load(f)
id_to_cat = {int(k): v for k, v in config['id_to_category'].items()}

# Load golden set
with open('data/gold_dataset/v6_golden_set.json') as f:
    golden = json.load(f)

print('=' * 60)
print('  v3c Golden Set 벤치마크 (전수 검수 227건)')
print('=' * 60)
print(f'\n  Golden set: {len(golden)}건 (전부 human_verified)')
print(f'  모델: {MODEL_DIR}')

# Golden set v3c 분포
true_dist = Counter(item['v3c_topic'] for item in golden)
print(f'\n  Golden v3c 정답 분포:')
for t in V3C_TOPICS:
    print(f'    {t}: {true_dist.get(t, 0)}건')

# Predict
y_true, y_pred = [], []
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
    margin = pred_conf - topk.values[1].item()
    pred_topic = id_to_cat[pred_id]
    top2_topic = id_to_cat[topk.indices[1].item()]

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

# Per-category P/R/F1
p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=V3C_TOPICS, zero_division=0)
per_cat = {}
for i, t in enumerate(V3C_TOPICS):
    per_cat[t] = {'precision': round(float(p[i]), 4), 'recall': round(float(r[i]), 4), 'f1': round(float(f1[i]), 4), 'support': int(support[i])}

# Comparison
print(f'\n{"="*60}')
print(f'  이전 최고 (v3c_full, 102건 golden): 85.3%')
print(f'  **v3c_full_clean (227건 golden):    {acc*100:.1f}%**')
print(f'  이전 대비: {(acc-0.853)*100:+.1f}%p')
print(f'{"="*60}')

# Save benchmark result
os.makedirs('models/v3/topic_v3c_full_clean', exist_ok=True)
with open('models/v3/topic_v3c_full_clean/golden_benchmark.json', 'w', encoding='utf-8') as f:
    json.dump({
        'golden_set_size': len(golden),
        'golden_human_verified': len(golden),
        'accuracy': acc,
        'correct': correct,
        'v3c_topics': V3C_TOPICS,
        'per_category': per_cat,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'misclassified': misclassified,
    }, f, ensure_ascii=False, indent=2)
print(f'\n  저장: models/v3/topic_v3c_full_clean/golden_benchmark.json')

# Update version_log
from datetime import datetime
log_path = 'benchmarks/version_log.json'
with open(log_path) as f:
    log = json.load(f)

previous = log["entries"][-1] if log["entries"] else None
previous_acc = previous["results"]["overall_accuracy"] if previous else None

new_entry = {
    "id": f"run_20260308_topic_v3c_full_clean",
    "timestamp": datetime.now().isoformat(),
    "model": {
        "name": "topic_v3c_full_clean",
        "path": "models/v3/topic_v3c_full_clean/final_model",
        "train_size": 9058,
        "epochs": 5
    },
    "golden_set": {
        "version": "v6",
        "total": 227,
        "human_verified": 227
    },
    "data_integrity": {
        "leak_check_passed": True,
        "golden_excluded": 172
    },
    "results": {
        "overall_accuracy": round(acc, 4),
        "per_category": per_cat,
        "misclassified_count": len(misclassified),
        "misclassified_details": misclassified[:20]
    },
    "comparison_to_previous": {
        "previous_id": previous["id"] if previous else None,
        "accuracy_delta": round(acc - previous_acc, 4) if previous_acc else None
    },
    "analysis": f"첫 벤치마크. 227건 전수 검수 golden set 기준 {acc*100:.1f}%. 병목: " +
        min(per_cat.items(), key=lambda x: x[1]['recall'])[0] +
        f" recall {min(per_cat.values(), key=lambda x: x['recall'])['recall']:.2f}",
    "next_direction": ""
}

# Set next_direction based on bottleneck
bottleneck_cat = min(per_cat.items(), key=lambda x: x[1]['recall'])
new_entry["next_direction"] = (
    f"병목 카테고리 '{bottleneck_cat[0]}' recall 개선 필요 "
    f"(현재 {bottleneck_cat[1]['recall']:.2f}). "
    f"소수 카테고리 학습 데이터 보강 또는 경계 규칙 조정 검토."
)

log["entries"].append(new_entry)
with open(log_path, 'w', encoding='utf-8') as f:
    json.dump(log, f, ensure_ascii=False, indent=2)
print(f'  version_log 업데이트: {log_path}')
