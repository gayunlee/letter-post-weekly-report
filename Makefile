.PHONY: setup pull push status

# 첫 세팅 (클론 후 1회)
setup:
	pip install boto3 python-dotenv
	python3 scripts/sync_data.py pull
	@echo "\n✅ 세팅 완료 — 데이터 다운로드됨"

# 데이터 최신화
pull:
	git pull
	python3 scripts/sync_data.py pull

# 데이터 업로드
push:
	python3 scripts/sync_data.py push

# 원격 파일 확인
status:
	python3 scripts/sync_data.py status
