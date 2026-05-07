"""BigQuery Writer — voc_labelled 데이터셋에 분류 결과 저장"""
import logging
from datetime import datetime
from google.cloud import bigquery

logger = logging.getLogger(__name__)

DATASET = "voc_labelled"

LETTERS_POSTS_SCHEMA = [
    bigquery.SchemaField("id", "STRING"),
    bigquery.SchemaField("source_type", "STRING"),  # letter / post
    bigquery.SchemaField("master_id", "STRING"),
    bigquery.SchemaField("master_name", "STRING"),
    bigquery.SchemaField("user_id", "STRING"),
    bigquery.SchemaField("content", "STRING"),
    bigquery.SchemaField("created_at", "TIMESTAMP"),
    bigquery.SchemaField("classified_at", "TIMESTAMP"),
    bigquery.SchemaField("topic", "STRING"),
    bigquery.SchemaField("subtag", "STRING"),
    bigquery.SchemaField("sentiment", "STRING"),
    bigquery.SchemaField("summary", "STRING"),
    bigquery.SchemaField("tags", "STRING", mode="REPEATED"),
    bigquery.SchemaField("confidence", "FLOAT"),
    bigquery.SchemaField("classifier_model", "STRING"),
    bigquery.SchemaField("pipeline_date", "DATE"),
]

CHANNEL_TALK_SCHEMA = [
    bigquery.SchemaField("chat_id", "STRING"),
    bigquery.SchemaField("text", "STRING"),
    bigquery.SchemaField("created_at", "TIMESTAMP"),
    bigquery.SchemaField("classified_at", "TIMESTAMP"),
    bigquery.SchemaField("topic", "STRING"),
    bigquery.SchemaField("confidence", "FLOAT"),
    bigquery.SchemaField("subtag", "STRING"),
    bigquery.SchemaField("source", "STRING"),  # kcelectra / llm_fallback
    bigquery.SchemaField("route", "STRING"),
    bigquery.SchemaField("classifier_model", "STRING"),
    bigquery.SchemaField("pipeline_date", "DATE"),
]


class BigQueryWriter:
    """voc_labelled에 분류 결과 저장"""

    def __init__(self, client: bigquery.Client):
        self.client = client
        self.project = client.project

    def _ensure_table(self, table_name, schema):
        """테이블이 없으면 생성 (DATE 파티션)"""
        table_id = f"{self.project}.{DATASET}.{table_name}"
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="pipeline_date",
        )
        table = self.client.create_table(table, exists_ok=True)
        return table_id

    def _delete_existing(self, table_id, pipeline_date):
        """동일 날짜 데이터 삭제 (멱등성)"""
        query = f"DELETE FROM `{table_id}` WHERE pipeline_date = '{pipeline_date}'"
        try:
            self.client.query(query).result()
            logger.info(f"  기존 데이터 삭제: {table_id} / {pipeline_date}")
        except Exception:
            pass  # 테이블이 비어있으면 무시

    def write_letters_posts(self, items, pipeline_date, classifier_model="bedrock-sonnet-3.7"):
        """편지글/게시글 분류 결과 저장"""
        table_id = self._ensure_table("letters_posts", LETTERS_POSTS_SCHEMA)
        self._delete_existing(table_id, pipeline_date)

        now = datetime.utcnow().isoformat()
        rows = []
        for item in items:
            cls = item.get("classification", {})
            content = item.get("message") or item.get("textBody") or item.get("body", "")
            rows.append({
                "id": item.get("_id", ""),
                "source_type": "letter" if "message" in item else "post",
                "master_id": item.get("masterId", ""),
                "master_name": item.get("masterName", ""),
                "user_id": item.get("userId", ""),
                "content": content[:5000],
                "created_at": item.get("createdAt", ""),
                "classified_at": now,
                "topic": cls.get("topic", ""),
                "subtag": cls.get("subtag", ""),
                "sentiment": cls.get("sentiment", ""),
                "summary": cls.get("summary", ""),
                "tags": cls.get("tags", []),
                "confidence": cls.get("confidence", 0.0),
                "classifier_model": classifier_model,
                "pipeline_date": pipeline_date,
            })

        if rows:
            errors = self.client.insert_rows_json(table_id, rows)
            if errors:
                logger.error(f"BigQuery 삽입 에러: {errors[:3]}")
                return 0
            logger.info(f"  letters_posts 저장: {len(rows)}건")

        return len(rows)

    def write_channel_talk(self, items, pipeline_date, classifier_model="kcelectra-v9 + bedrock-haiku-4.5"):
        """채널톡 분류 결과 저장"""
        table_id = self._ensure_table("channel_talk", CHANNEL_TALK_SCHEMA)
        self._delete_existing(table_id, pipeline_date)

        now = datetime.utcnow().isoformat()
        rows = []
        for item in items:
            cls = item.get("classification", {})
            ts = item.get("first_message_at")
            if isinstance(ts, (int, float)):
                ts = datetime.utcfromtimestamp(ts / 1000).isoformat()
            elif not ts:
                ts = None
            rows.append({
                "chat_id": item.get("chatId", ""),
                "text": item.get("text", "")[:5000],
                "created_at": ts,
                "classified_at": now,
                "topic": cls.get("topic", ""),
                "confidence": cls.get("confidence", 0.0),
                "subtag": cls.get("subtag", ""),
                "source": cls.get("source", ""),
                "route": item.get("route", ""),
                "classifier_model": classifier_model,
                "pipeline_date": pipeline_date,
            })

        if rows:
            errors = self.client.insert_rows_json(table_id, rows)
            if errors:
                logger.error(f"BigQuery 삽입 에러: {errors[:3]}")
                return 0
            logger.info(f"  channel_talk 저장: {len(rows)}건")

        return len(rows)
