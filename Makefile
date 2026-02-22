.PHONY: setup pull push status

# 첫 세팅 (클론 후 1회)
setup:
	pip install boto3 python-dotenv
	python scripts/sync_data.py pull
	@echo "\n✅ 세팅 완료 — 데이터 다운로드됨"

# 데이터 최신화
pull:
	git pull
	python scripts/sync_data.py pull

# 데이터 업로드
push:
	python scripts/sync_data.py push

# 원격 파일 확인
status:
	python scripts/sync_data.py status
