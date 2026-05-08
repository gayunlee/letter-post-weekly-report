# Letter/Post Weekly Report

VOC classification and reporting pipeline for US service data.

This project classifies customer feedback from letters, posts, and ChannelTalk,
stores labelled rows in BigQuery, and generates reports for Notion/Slack.

## What It Does

- Classifies letters and posts with Bedrock.
- Classifies ChannelTalk conversations with the ChannelTalk classifier path.
- Writes labelled data to BigQuery `voc_labelled`.
- Generates weekly reports from labelled BigQuery tables.
- Publishes reports to Notion and Slack in the scheduled production path.

## Main Data Flow

```text
Source data
  ├─ Letters/posts
  └─ Channel.io messages/chats

Pipeline
  ├─ scripts/run_daily_pipeline.py
  │   ├─ letters/posts classification
  │   └─ ChannelTalk classification
  └─ scripts/generate_weekly_report_v5.py

Output
  ├─ BigQuery: voc_labelled.letters_posts
  ├─ BigQuery: voc_labelled.channel_talk
  ├─ Notion report page
  └─ Slack notification
```

## Repository Map

| Path | Purpose |
|---|---|
| `scripts/run_daily_pipeline.py` | Daily classification pipeline |
| `scripts/generate_weekly_report_v5.py` | Weekly report generation |
| `src/bigquery/` | BigQuery clients, queries, and writers |
| `src/classifier_v5/` | Letters/posts Bedrock classifier |
| `src/classifier_v4/` | ChannelTalk classifier path |
| `src/reporter/` | Report analytics and rendering helpers |
| `src/integrations/` | Notion and Slack integrations |
| `dashboard/` | Streamlit dashboard apps |
| `deploy/` | Cloud Run Jobs, Scheduler, and dashboard deployment scripts |
| `tests/` | Unit tests |

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` from `.env.example` and configure credentials for:

- Google Cloud / BigQuery
- AWS Bedrock
- Notion
- Slack

Do not commit `.env`, service account keys, or downloaded credentials.

## Common Commands

Run the unit tests:

```bash
python3 -m unittest discover tests
```

Run the daily pipeline for a specific date:

```bash
python3 scripts/run_daily_pipeline.py --date YYYY-MM-DD --skip-slack
```

Run a dry run:

```bash
python3 scripts/run_daily_pipeline.py --date YYYY-MM-DD --dry-run --skip-slack
```

Run ChannelTalk only for one date:

```bash
python3 scripts/run_daily_pipeline.py --date YYYY-MM-DD --skip-letters-posts --skip-slack
```

`--skip-letters` is an alias for `--skip-letters-posts`.

Generate a weekly report:

```bash
python3 scripts/generate_weekly_report_v5.py --start YYYY-MM-DD --end YYYY-MM-DD
```

`--end` is exclusive.

## Scheduled Production Jobs

Production scheduling is handled with Cloud Run Jobs and Cloud Scheduler.
See [deploy/README.md](deploy/README.md) for detailed deployment and operations.

Main jobs:

| Job | Purpose |
|---|---|
| `voc-daily` | Classifies the daily data and writes to `voc_labelled` |
| `voc-weekly` | Reads `voc_labelled`, generates the weekly report, and publishes it |

Important date convention:

- The default daily target date is **yesterday in KST**.
- Example: a run at `2026-05-08 08:00 KST` processes `2026-05-07 00:00~23:59 KST`.
- Avoid manually running same-day data unless intentionally testing partial data.

## ChannelTalk Notes

The upstream Channel.io tables can contain multiple channels.
The current VOC pipeline scopes ChannelTalk reads to `us-plus` by default.

Operational implications:

- ChannelTalk reingest reads `us-plus` by default.
- ChannelTalk reingest deletes and reinserts the target `pipeline_date` in
  `voc_labelled.channel_talk`.
- Use `--skip-letters-posts` when reingesting ChannelTalk only.
- Do not operate `us-campus` in the same labelled table until channel columns
  and channel-scoped deletes are supported end to end.

## BigQuery Output Tables

| Table | Description |
|---|---|
| `voc_labelled.letters_posts` | Classified letters and posts |
| `voc_labelled.channel_talk` | Classified ChannelTalk conversations |

Both tables are partitioned by `pipeline_date`.

## Deployment

Build and push the Cloud Run image:

```bash
gcloud builds submit --config=deploy/cloudbuild.yaml .
```

Deploy or update Cloud Run Jobs and Scheduler:

```bash
bash deploy/deploy_jobs.sh
```

Dashboard deployment is separate:

```bash
bash deploy/deploy_dashboard.sh
```

## Troubleshooting

Check recent Cloud Run executions:

```bash
gcloud beta run jobs executions list --job=voc-daily --region=asia-northeast3
```

Check logs for an execution:

```bash
gcloud beta run jobs executions logs EXECUTION_ID --region=asia-northeast3
```

Check BigQuery counts for a date:

```sql
SELECT pipeline_date, COUNT(*) AS rows
FROM `us-service-data.voc_labelled.channel_talk`
WHERE pipeline_date = DATE 'YYYY-MM-DD'
GROUP BY pipeline_date;
```

Common failure surfaces:

- Missing local or Secret Manager credentials.
- BigQuery permissions for the Cloud Run service account.
- AWS Bedrock throttling or credential failure.
- Notion/Slack integration token or database/channel permission issues.
- Running a date before the KST day has fully completed.
