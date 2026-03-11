"""v3c Full Data + Class Weight 모델 Golden 벤치마크"""
import sys, os, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

MODEL_DIR = 'models/v3/topic_v3c_full/final_model'
V3C_TOPICS = ['운영 피드백', '서비스 피드백', '콘텐츠·투자', '일상·감사', '기타']

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(f'{MODEL_DIR}/category_config.json') as f:
    config = json.load(f)
id_to_cat = {int(k): v for k, v in config['id_to_category'].items()}

with open('data/gold_dataset/v6_golden_set.json') as f:
    golden = json.load(f)

y_true, y_pred = [], []
human_true, human_pred = [], []
auto_true, auto_pred = [], []
misclassified = []

for item in golden:
    inputs = tokenizer(item['text'], truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    topk = torch.topk(probs, k=2)
    pred_id = topk.indices[0].item()
    pred_conf = topk.values[0].item()
    margin = pred_conf - topk.values[1].item()

    pred = id_to_cat[pred_id]
    true = item['v3c_topic']

    y_true.append(true)
    y_pred.append(pred)

    if item.get('_golden_source') == 'human_verified':
        human_true.append(true)
        human_pred.append(pred)
    else:
        auto_true.append(true)
        auto_pred.append(pred)

    if true != pred:
        misclassified.append({
            'text': item['text'][:70].replace('\n', ' '),
            'true': true, 'pred': pred,
            'conf': round(pred_conf, 3), 'margin': round(margin, 3),
            'source': 'H' if item.get('_golden_source') == 'human_verified' else 'A'
        })

acc = accuracy_score(y_true, y_pred)
h_acc = accuracy_score(human_true, human_pred)
a_acc = accuracy_score(auto_true, auto_pred)

print('=' * 60)
print('  v3c Full Data + Class Weight 모델 Golden 벤치마크')
print('=' * 60)
print(f'  전체: {acc*100:.1f}% ({sum(1 for a, b in zip(y_true, y_pred) if a == b)}/{len(golden)})')
print(f'  사람 검수 (102건): {h_acc*100:.1f}%')
print(f'  자동 검수 (125건): {a_acc*100:.1f}%')
print()
print(classification_report(y_true, y_pred, labels=V3C_TOPICS, digits=3, zero_division=0))

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

print(f'\n  오분류: {len(misclassified)}건')
for m in misclassified:
    print(f"    [{m['source']}] [{m['true']} -> {m['pred']}] conf={m['conf']:.2f} margin={m['margin']:.2f} | {m['text'][:55]}")

print(f'\n{"="*60}')
print(f'  v2 (구):                  69.3%')
print(f'  v3c 5,962건 편향:         75.3% (227건)')
print(f'  v3c 5,962건 편향 102건:   85.3%')
print(f'  v3c 3,754건 균형:         66.5% (227건)')
print(f'  **v3c 13,995건+weight:   {acc*100:.1f}% (227건)**')
print(f'  **v3c 13,995건+weight 102건: {h_acc*100:.1f}%**')
print(f'{"="*60}')
