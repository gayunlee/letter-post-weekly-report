"""서브 테마 분석 모듈 테스트"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classifier_v2.sub_theme_analyzer import SubThemeAnalyzer, save_sub_themes

DATA_PATH = "./data/classified_data_two_axis/2026-02-02.json"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"데이터 파일 없음: {DATA_PATH}")
        return

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    letters = data.get("letters", [])
    posts = data.get("posts", [])
    print(f"편지 {len(letters)}건, 게시글 {len(posts)}건 로드")

    analyzer = SubThemeAnalyzer()
    result = analyzer.analyze(letters, posts)

    # 서비스 이슈 클러스터
    clusters = result.get("service_clusters", {})
    print(f"\n서비스 이슈 클러스터: {len(clusters)}개")
    for cid, info in sorted(clusters.items(), key=lambda x: x[1]["count"], reverse=True):
        masters = ", ".join(f"{m}({c})" for m, c in info.get("top_masters", [])[:2])
        print(f"  [{info['label']}] {info['count']}건 — {masters}")

    # 특이 패턴
    patterns = result.get("notable_patterns", [])
    print(f"\n특이 패턴: {len(patterns)}개")
    for p in patterns:
        print(f"  [{p['topic']}] 부정 {p['negative_count']}/{p['total_in_topic']}건 ({p['negative_ratio']}%)")
        print(f"    {p['summary'][:200]}")

    # 저장 테스트
    path = save_sub_themes(result, "2026-02-02")
    print(f"\n저장 완료: {path}")

if __name__ == "__main__":
    main()
