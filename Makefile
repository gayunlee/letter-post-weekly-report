.PHONY: setup pull push status extract-v3-model classify-v3

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

# v3 모델 추출 (Colab 결과 zip → models/v3/topic/final_model/)
extract-v3-model:
	@if [ ! -f data/colab_export/v3_topic_model.zip ]; then \
		echo "v3_topic_model.zip이 없습니다. make pull을 먼저 실행하세요."; \
		exit 1; \
	fi
	mkdir -p models/v3/topic/final_model
	unzip -o data/colab_export/v3_topic_model.zip -d models/v3/topic/final_model
	@echo "\n모델 추출 완료: models/v3/topic/final_model/"
	@ls models/v3/topic/final_model/

# v3 분류 실행
classify-v3:
	python3 scripts/classify_v3.py
