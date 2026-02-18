"""파이프라인 벤치마크 스크립트

분류 정확도와 각 단계별 시간을 측정하고 결과를 누적 저장합니다.

사용법:
    # 분류 벤치마크 (기본)
    python scripts/benchmark.py

    # 리포트 생성 시간까지 측정 (Claude API 비용 발생)
    python scripts/benchmark.py --with-report

    # 특정 분류기만 테스트
    python scripts/benchmark.py --classifier vector
    python scripts/benchmark.py --classifier finetuned

    # 이전 결과와 비교
    python scripts/benchmark.py --compare

    # 메모 추가 (어떤 변경 후 측정인지 기록)
    python scripts/benchmark.py --note "한국어 임베딩 모델로 교체"
"""
import json
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
BENCHMARKS_DIR.mkdir(exist_ok=True)

# 기존 → 새 카테고리 매핑
OLD_TO_NEW = {
    "감사·후기": "긍정 피드백",
    "질문·토론": "질문/문의",
    "정보성 글": "정보 공유",
    "서비스 피드백": "부정 피드백",
    "불편사항": "부정 피드백",
    "일상·공감": "일상 소통",
    "서비스 불편사항": "부정 피드백",
    "서비스 제보/건의": "부정 피드백",
}

CATEGORIES_NEW = ["긍정 피드백", "부정 피드백", "질문/문의", "정보 공유", "일상 소통"]


def load_test_data(data_file: Path) -> tuple:
    """분류 대상 데이터 로드 (분류 결과 제거)"""
    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)

    letters = data.get("letters", [])
    posts = data.get("posts", [])

    # 기존 분류 결과 제거 (재분류 위해)
    for item in letters + posts:
        item.pop("classification", None)

    return letters, posts


def load_ground_truth(labeling_file: Path) -> Dict[str, str]:
    """Ground truth 로드 → {id: category}"""
    if not labeling_file.exists():
        return {}

    with open(labeling_file, encoding="utf-8") as f:
        data = json.load(f)

    gt = {}
    for item in data:
        item_id = item.get("id")
        label = item.get("new_label")
        if item_id and label:
            gt[item_id] = label
    return gt


def map_category(category: str, scheme: str) -> str:
    """카테고리를 새 스키마로 매핑 (old→new). scheme이 new면 그대로."""
    if scheme == "old":
        return OLD_TO_NEW.get(category, category)
    return category


def compute_metrics(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict[str, Any]:
    """정확도, precision, recall, F1, confusion matrix 계산"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_category = {}
    for i, cat in enumerate(labels):
        per_category[cat] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    return {
        "accuracy": round(float(acc), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "per_category": per_category,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
    }


def benchmark_vector_classifier(
    letters: List[Dict], posts: List[Dict], use_llm_fallback: bool = False
) -> Dict[str, Any]:
    """벡터 분류기 벤치마크"""
    from src.classifier.vector_classifier import VectorContentClassifier

    classifier = VectorContentClassifier(
        use_llm_fallback=use_llm_fallback,
        confidence_threshold=0.3,
    )

    start = time.time()
    classified_letters = classifier.classify_batch(
        [l.copy() for l in letters], content_field="message"
    )
    classified_posts = classifier.classify_batch(
        [p.copy() for p in posts], content_field="textBody"
    )
    elapsed = time.time() - start

    total = len(letters) + len(posts)
    all_items = classified_letters + classified_posts

    # confidence 분포
    confidences = [
        item.get("classification", {}).get("confidence", 0) for item in all_items
    ]

    return {
        "name": "vector" + ("+llm" if use_llm_fallback else ""),
        "category_scheme": "old",
        "elapsed_seconds": round(elapsed, 3),
        "time_per_item_ms": round(elapsed / total * 1000, 3) if total else 0,
        "total_items": total,
        "llm_fallback_calls": classifier.llm_fallback_count,
        "classified_items": all_items,
        "category_distribution": dict(
            Counter(
                item.get("classification", {}).get("category", "미분류")
                for item in all_items
            )
        ),
        "confidence_stats": {
            "mean": round(float(np.mean(confidences)), 4),
            "median": round(float(np.median(confidences)), 4),
            "min": round(float(np.min(confidences)), 4),
            "p25": round(float(np.percentile(confidences, 25)), 4),
            "p75": round(float(np.percentile(confidences, 75)), 4),
        },
    }


def benchmark_finetuned_classifier(
    letters: List[Dict], posts: List[Dict]
) -> Dict[str, Any]:
    """파인튜닝 분류기 벤치마크"""
    from src.classifier_v2.finetuned_classifier import FinetunedClassifier

    classifier = FinetunedClassifier()

    start = time.time()
    classified_letters = classifier.classify_batch(
        [l.copy() for l in letters], content_field="message"
    )
    classified_posts = classifier.classify_batch(
        [p.copy() for p in posts], content_field="textBody"
    )
    elapsed = time.time() - start

    total = len(letters) + len(posts)
    all_items = classified_letters + classified_posts

    confidences = [
        item.get("classification", {}).get("confidence", 0) for item in all_items
    ]

    return {
        "name": "finetuned",
        "category_scheme": "old",
        "elapsed_seconds": round(elapsed, 3),
        "time_per_item_ms": round(elapsed / total * 1000, 3) if total else 0,
        "total_items": total,
        "llm_fallback_calls": 0,
        "classified_items": all_items,
        "category_distribution": dict(
            Counter(
                item.get("classification", {}).get("category", "미분류")
                for item in all_items
            )
        ),
        "confidence_stats": {
            "mean": round(float(np.mean(confidences)), 4),
            "median": round(float(np.median(confidences)), 4),
            "min": round(float(np.min(confidences)), 4),
            "p25": round(float(np.percentile(confidences, 25)), 4),
            "p75": round(float(np.percentile(confidences, 75)), 4),
        },
    }


def benchmark_report_generation(
    letters: List[Dict], posts: List[Dict],
    prev_letters: Optional[List[Dict]] = None,
    prev_posts: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """리포트 생성 벤치마크 (analytics + report gen)"""
    from src.reporter.analytics import WeeklyAnalytics
    from src.reporter.report_generator import ReportGenerator

    analytics = WeeklyAnalytics()

    # Analytics 단계
    start_analytics = time.time()
    stats = analytics.analyze_weekly_data(
        letters, posts,
        previous_letters=prev_letters,
        previous_posts=prev_posts,
    )
    analytics_time = time.time() - start_analytics

    # 활성 마스터 수
    active_masters = sum(
        1 for m in stats["master_stats"].values()
        if m["this_week"]["total"] > 0
    )

    # Report generation 단계
    generator = ReportGenerator()
    start_report = time.time()
    report = generator.generate_report(
        stats,
        start_date="2026-01-05",
        end_date="2026-01-12",
    )
    report_time = time.time() - start_report

    return {
        "analytics_time_seconds": round(analytics_time, 3),
        "report_gen_time_seconds": round(report_time, 3),
        "total_time_seconds": round(analytics_time + report_time, 3),
        "active_masters": active_masters,
        "api_calls_estimated": 1 + active_masters,  # 1 summary + N masters
        "report_length_chars": len(report),
    }


def evaluate_accuracy(
    classifier_result: Dict[str, Any],
    ground_truth: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """분류 결과를 ground truth와 비교하여 정확도 측정"""
    items = classifier_result["classified_items"]
    scheme = classifier_result["category_scheme"]

    y_true = []
    y_pred = []

    for item in items:
        item_id = item.get("_id")
        if item_id and item_id in ground_truth:
            true_label = ground_truth[item_id]
            pred_label = map_category(
                item.get("classification", {}).get("category", "미분류"),
                scheme,
            )
            # ground truth에 있는 카테고리만 평가
            if true_label in CATEGORIES_NEW:
                y_true.append(true_label)
                y_pred.append(pred_label)

    if not y_true:
        return None

    metrics = compute_metrics(y_true, y_pred, CATEGORIES_NEW)
    metrics["evaluated_items"] = len(y_true)
    return metrics


def save_result(result: Dict[str, Any]) -> Path:
    """벤치마크 결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = BENCHMARKS_DIR / f"{timestamp}.json"

    # classified_items는 용량이 크므로 저장에서 제외
    save_data = json.loads(json.dumps(result, default=str))
    for clf in save_data.get("classifiers", {}).values():
        clf.pop("classified_items", None)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    return filepath


def load_previous_result() -> Optional[Dict[str, Any]]:
    """가장 최근 벤치마크 결과 로드"""
    results = sorted(BENCHMARKS_DIR.glob("*.json"))
    if not results:
        return None

    with open(results[-1], encoding="utf-8") as f:
        return json.load(f)


def print_classification_results(name: str, result: Dict[str, Any], metrics: Optional[Dict[str, Any]]):
    """분류 벤치마크 결과 출력"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  처리 건수:     {result['total_items']:,}건")
    print(f"  처리 시간:     {result['elapsed_seconds']:.3f}초")
    print(f"  건당 시간:     {result['time_per_item_ms']:.3f}ms")
    if result["llm_fallback_calls"] > 0:
        print(f"  LLM 호출:      {result['llm_fallback_calls']}회")

    print(f"\n  Confidence 분포:")
    cs = result["confidence_stats"]
    print(f"    평균: {cs['mean']:.4f}  |  중앙값: {cs['median']:.4f}  |  최소: {cs['min']:.4f}")
    print(f"    25%: {cs['p25']:.4f}   |  75%: {cs['p75']:.4f}")

    print(f"\n  카테고리 분포:")
    for cat, cnt in sorted(result["category_distribution"].items(), key=lambda x: -x[1]):
        pct = cnt / result["total_items"] * 100
        bar = "█" * int(pct / 2)
        print(f"    {cat:12s}  {cnt:5d} ({pct:5.1f}%) {bar}")

    if metrics:
        print(f"\n  정확도 (Ground Truth {metrics['evaluated_items']}건 기준):")
        print(f"    Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")

        print(f"\n  카테고리별 성능:")
        print(f"    {'카테고리':12s}  {'Precision':>9s}  {'Recall':>7s}  {'F1':>7s}  {'건수':>5s}")
        print(f"    {'-'*48}")
        for cat in CATEGORIES_NEW:
            m = metrics["per_category"].get(cat, {})
            if m.get("support", 0) > 0:
                print(f"    {cat:12s}  {m['precision']:9.4f}  {m['recall']:7.4f}  {m['f1']:7.4f}  {m['support']:5d}")


def print_report_results(result: Dict[str, Any]):
    """리포트 생성 벤치마크 결과 출력"""
    print(f"\n{'='*60}")
    print(f"  리포트 생성")
    print(f"{'='*60}")
    print(f"  Analytics 시간:    {result['analytics_time_seconds']:.3f}초")
    print(f"  리포트 생성 시간:  {result['report_gen_time_seconds']:.3f}초")
    print(f"  총 시간:           {result['total_time_seconds']:.3f}초")
    print(f"  활성 마스터 수:    {result['active_masters']}명")
    print(f"  예상 API 호출:     {result['api_calls_estimated']}회")
    print(f"  리포트 크기:       {result['report_length_chars']:,}자")


def print_comparison(current: Dict, previous: Dict):
    """이전 결과와 비교 출력"""
    print(f"\n{'='*60}")
    print(f"  이전 결과 비교")
    print(f"{'='*60}")
    print(f"  이전: {previous.get('timestamp', '?')}  |  현재: {current.get('timestamp', '?')}")

    if previous.get("note"):
        print(f"  이전 메모: {previous['note']}")
    if current.get("note"):
        print(f"  현재 메모: {current['note']}")

    # 분류기별 비교
    for clf_name in current.get("classifiers", {}):
        curr_clf = current["classifiers"][clf_name]
        prev_clf = previous.get("classifiers", {}).get(clf_name)
        if not prev_clf:
            continue

        print(f"\n  [{clf_name}]")

        # 시간 비교
        curr_time = curr_clf["elapsed_seconds"]
        prev_time = prev_clf["elapsed_seconds"]
        time_diff = curr_time - prev_time
        time_pct = (time_diff / prev_time * 100) if prev_time else 0
        arrow = "↓" if time_diff < 0 else "↑" if time_diff > 0 else "→"
        print(f"    시간: {prev_time:.3f}초 → {curr_time:.3f}초  ({arrow} {abs(time_pct):.1f}%)")

        # 정확도 비교
        curr_acc = curr_clf.get("accuracy", {}).get("accuracy")
        prev_acc = prev_clf.get("accuracy", {}).get("accuracy")
        if curr_acc is not None and prev_acc is not None:
            acc_diff = curr_acc - prev_acc
            arrow = "↑" if acc_diff > 0 else "↓" if acc_diff < 0 else "→"
            print(f"    정확도: {prev_acc:.4f} → {curr_acc:.4f}  ({arrow} {abs(acc_diff)*100:.1f}%p)")

            curr_f1 = curr_clf.get("accuracy", {}).get("f1_weighted", 0)
            prev_f1 = prev_clf.get("accuracy", {}).get("f1_weighted", 0)
            f1_diff = curr_f1 - prev_f1
            arrow = "↑" if f1_diff > 0 else "↓" if f1_diff < 0 else "→"
            print(f"    F1:     {prev_f1:.4f} → {curr_f1:.4f}  ({arrow} {abs(f1_diff)*100:.1f}%p)")

    # 리포트 생성 비교
    curr_rpt = current.get("report_generation")
    prev_rpt = previous.get("report_generation")
    if curr_rpt and prev_rpt:
        print(f"\n  [리포트 생성]")
        curr_t = curr_rpt["total_time_seconds"]
        prev_t = prev_rpt["total_time_seconds"]
        diff = curr_t - prev_t
        arrow = "↓" if diff < 0 else "↑" if diff > 0 else "→"
        print(f"    시간: {prev_t:.3f}초 → {curr_t:.3f}초  ({arrow} {abs(diff/prev_t*100):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="파이프라인 벤치마크")
    parser.add_argument(
        "--classifier",
        choices=["vector", "finetuned", "all"],
        default="all",
        help="테스트할 분류기 (기본: all)",
    )
    parser.add_argument(
        "--with-report",
        action="store_true",
        help="리포트 생성 시간도 측정 (Claude API 비용 발생)",
    )
    parser.add_argument(
        "--with-llm-fallback",
        action="store_true",
        help="벡터 분류기의 LLM fallback 활성화",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="테스트 데이터 파일 (기본: 가장 최근 파일)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="이전 결과와 비교만 표시",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="이 벤치마크에 대한 메모",
    )
    args = parser.parse_args()

    # 비교 모드
    if args.compare:
        results = sorted(BENCHMARKS_DIR.glob("*.json"))
        if len(results) < 2:
            print("비교할 결과가 2개 이상 필요합니다.")
            return
        with open(results[-2], encoding="utf-8") as f:
            prev = json.load(f)
        with open(results[-1], encoding="utf-8") as f:
            curr = json.load(f)
        print_comparison(curr, prev)
        return

    # 데이터 파일 결정
    if args.data_file:
        data_file = Path(args.data_file)
    else:
        # 가장 최근 파일 사용
        data_files = sorted((PROJECT_ROOT / "data" / "classified_data").glob("*.json"))
        data_file = data_files[-1] if data_files else None

    if not data_file or not data_file.exists():
        print(f"데이터 파일을 찾을 수 없습니다: {data_file}")
        return

    # Ground truth 로드
    gt_file = PROJECT_ROOT / "data" / "labeling" / "refined_labeled.json"
    ground_truth = load_ground_truth(gt_file)

    print("=" * 60)
    print("  파이프라인 벤치마크")
    print("=" * 60)
    print(f"  데이터: {data_file.name}")
    if args.note:
        print(f"  메모: {args.note}")

    # 데이터 로드
    letters, posts = load_test_data(data_file)
    total = len(letters) + len(posts)
    print(f"  편지: {len(letters)}건  |  게시글: {len(posts)}건  |  총: {total}건")

    # ground truth 매칭 수 확인
    all_ids = set(item.get("_id") for item in letters + posts)
    gt_overlap = len(set(ground_truth.keys()) & all_ids)
    print(f"  Ground Truth 매칭: {gt_overlap}건 / {len(ground_truth)}건")

    if gt_overlap == 0:
        print("\n  ⚠ Ground Truth 매칭이 0건입니다. 정확도 측정 불가.")
        print(f"    데이터 파일의 ID와 라벨링 데이터의 ID가 일치하지 않습니다.")
        print(f"    --data-file 옵션으로 매칭되는 데이터를 지정하세요.")

    # 결과 저장용
    result = {
        "timestamp": datetime.now().isoformat(),
        "note": args.note,
        "data": {
            "source": data_file.name,
            "total_items": total,
            "letters": len(letters),
            "posts": len(posts),
            "ground_truth_file": gt_file.name,
            "ground_truth_matched": gt_overlap,
        },
        "classifiers": {},
        "report_generation": None,
    }

    # 분류기 벤치마크
    classifiers_to_run = []
    if args.classifier in ("vector", "all"):
        classifiers_to_run.append("vector")
    if args.classifier in ("finetuned", "all"):
        classifiers_to_run.append("finetuned")

    for clf_name in classifiers_to_run:
        print(f"\n>>> {clf_name} 분류기 실행 중...")

        try:
            if clf_name == "vector":
                clf_result = benchmark_vector_classifier(
                    letters, posts, use_llm_fallback=args.with_llm_fallback
                )
            else:
                clf_result = benchmark_finetuned_classifier(letters, posts)

            # 정확도 평가
            metrics = None
            if gt_overlap > 0:
                metrics = evaluate_accuracy(clf_result, ground_truth)

            print_classification_results(clf_name, clf_result, metrics)

            # 저장용 데이터 (classified_items 제외)
            save_clf = {k: v for k, v in clf_result.items() if k != "classified_items"}
            save_clf["accuracy"] = metrics
            result["classifiers"][clf_name] = save_clf

            # --with-report인 경우 분류된 데이터 보존
            if args.with_report and clf_name == classifiers_to_run[0]:
                report_items = clf_result["classified_items"]

        except Exception as e:
            print(f"  ✗ {clf_name} 실행 실패: {e}")
            import traceback
            traceback.print_exc()

    # 리포트 생성 벤치마크
    if args.with_report:
        print(f"\n>>> 리포트 생성 벤치마크 실행 중...")
        try:
            # 첫 번째 분류기 결과로 리포트 생성
            classified_letters = [
                item for item in report_items
                if item.get("type") == "LETTER" or "message" in item
            ]
            classified_posts = [
                item for item in report_items
                if item not in classified_letters
            ]

            rpt_result = benchmark_report_generation(
                classified_letters, classified_posts
            )
            print_report_results(rpt_result)
            result["report_generation"] = rpt_result
        except Exception as e:
            print(f"  ✗ 리포트 생성 실패: {e}")
            import traceback
            traceback.print_exc()

    # 결과 저장
    filepath = save_result(result)
    print(f"\n>>> 결과 저장: {filepath}")

    # 이전 결과 비교
    prev_results = sorted(BENCHMARKS_DIR.glob("*.json"))
    if len(prev_results) >= 2:
        with open(prev_results[-2], encoding="utf-8") as f:
            prev = json.load(f)
        print_comparison(result, prev)

    print("\n완료!")


if __name__ == "__main__":
    main()
