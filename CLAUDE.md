# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **weekly report generation system** that analyzes user-generated content (letters and posts) from a financial content platform. The system queries BigQuery, classifies content using AI/ML techniques, and generates comprehensive weekly insight reports for platform operators.

### Key Concepts

- **Masters (마스터)**: Financial content creators who run investment education communities
- **Subscribers (구독자)**: Users who subscribe to masters' content
- **Letters (편지글)**: Private messages sent to masters
- **Posts (게시글)**: Public board posts within each master's community
- **Official Clubs (오피셜클럽)**: Individual communities run by each master

Content includes investment questions, emotional support messages, user-to-user information sharing, service improvement suggestions, complaints, and feedback.

## Data Architecture

### BigQuery Access
- Authentication: `accountKey.json` (service account key file)
- The BigQuery tables contain letter and post data with metadata about masters, subscribers, timestamps, and content
- Data needs to be queried by weekly date ranges for report generation

### Content Classification System
The system must classify content into categories such as:
- Service improvement suggestions (서비스개선점)
- Informational posts (정보성 글)
- Gratitude/testimonials (감사·후기)
- Investment questions (투자 질문)
- Complaints (불편사항)
- User-to-user information sharing (정보공유)

Classification approach:
1. Query weekly data from BigQuery
2. Vectorize content for semantic search/classification
3. Use few-shot learning with examples from `example.md` to guide classification
4. Label data based on classification results
5. Generate statistics and insights

## Report Format

The output report must follow the structure in `example.md`:

### Required Sections
1. **핵심 요약 (Executive Summary)**: Overall metrics table with week-over-week comparison
2. **오피셜클럽별 상세 (Details by Official Club)**: Per-master breakdowns including:
   - Letter/post count table with previous week comparison
   - 주요 내용 (Key Content): Representative quotes and themes
   - 플랫폼/서비스 피드백 (Platform/Service Feedback): Service improvement opportunities
   - 체크 포인트 (Check Points): Action items and observations

### Key Metrics to Track
- Total letter count (편지 건수)
- Total post count (게시글 건수)
- Week-over-week changes (증감)
- Per-master breakdowns
- Content category distributions

### Report Style Guidelines
- Use Korean language throughout
- Include direct quotes from user content in italics (e.g., _"quote here"_)
- Provide week-over-week comparison tables
- Highlight actionable insights with arrow indicators (_→ recommendation_)
- Use markdown tables for statistics
- Group masters by activity level (main masters vs. "기타 마스터")

## Development Workflow

### Initial Setup
```bash
# Install dependencies (to be created)
pip install -r requirements.txt  # if exists

# Set up BigQuery authentication
export GOOGLE_APPLICATION_CREDENTIALS="./accountKey.json"
```

### Data Query Pattern
1. Define date ranges for current week and previous week
2. Query both letter and post tables from BigQuery
3. Retrieve text content along with metadata (master_id, created_at, etc.)
4. Sample data first to understand schema before full queries

### Classification Pipeline
1. Load sample data and `example.md` to establish classification patterns
2. Implement vectorization (consider using embeddings from OpenAI, Anthropic, or open-source models)
3. Use few-shot prompting with examples to classify content
4. Store vectors and labels for efficient reprocessing
5. Generate statistics from labeled data

### Report Generation
1. Aggregate classified data by master and category
2. Calculate week-over-week metrics
3. Extract representative quotes for each category
4. Identify recurring themes/FAQ items
5. Format output matching `example.md` structure

## Important Constraints

### Data Privacy
- `accountKey.json` contains sensitive credentials - never commit changes to this file
- User content may contain personal information - handle appropriately
- Generated reports are for internal platform use

### Accuracy Requirements
- Statistics must be precise and verifiable
- Classification should be consistent across similar content
- Week-over-week comparisons must use exact date ranges
- Representative quotes must be actual user content (no fabrication)

### Classification Quality
- Use few-shot learning to improve classification accuracy
- Vectorization enables semantic similarity matching
- Consider edge cases where content fits multiple categories
- Validate classification results against example report patterns

## Technical Considerations

When implementing this system:
- BigQuery queries should be efficient and avoid unnecessary data transfer
- Vector storage should support incremental updates for new weekly data
- Classification prompts should reference `example.md` patterns
- Report generation should be reproducible with same input data
- Consider caching/storing vectors to avoid reprocessing historical data

## File References

- `requirements.md`: Original Korean requirements document (lines 1-37)
- `example.md`: Complete example report for 2025.12.22-28 week (lines 1-575)
- `accountKey.json`: BigQuery service account credentials (DO NOT MODIFY)
