"""리포트만 재생성하는 스크립트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.storage.data_store import ClassifiedDataStore
from src.reporter.analytics import WeeklyAnalytics
from src.reporter.report_generator import ReportGenerator

print("데이터 로드 중...", flush=True)
data_store = ClassifiedDataStore()

this_week = data_store.load_weekly_data('2025-12-29')
prev_week = data_store.load_weekly_data('2025-12-22')
print("✓ 데이터 로드 완료", flush=True)

print("통계 분석 중...", flush=True)
analytics = WeeklyAnalytics()
stats = analytics.analyze_weekly_data(
    this_week['letters'], this_week['posts'],
    prev_week['letters'], prev_week['posts']
)
print(f"✓ 통계 분석 완료 (마스터 {len(stats['master_stats'])}개)", flush=True)

print("리포트 생성 중...", flush=True)
generator = ReportGenerator()
report = generator.generate_report(
    stats, '2025-12-29', '2026-01-05',
    output_path='./reports/weekly_report_2025-12-29.md'
)
print("✓ 리포트 생성 완료", flush=True)
print("저장 위치: ./reports/weekly_report_2025-12-29.md")
