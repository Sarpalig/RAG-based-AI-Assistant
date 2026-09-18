# Report Figure Assets

This folder contains reproducible assets for the university internship report.

## Setup

Install the normal application dependencies first:

```powershell
python -m pip install -r requirements.txt
```

Install the report-only screenshot and plotting dependencies:

```powershell
python -m pip install -r requirements-report.txt
python -m playwright install chromium
```

## Generate All Figures

```powershell
python scripts/generate_report_figures.py
```

The script starts the Streamlit app automatically in isolated report mode by setting `REPORT_FIGURE_MODE=1`. Normal application behavior is unchanged when this variable is not set. Ollama or OpenRouter is not required for report-mode screenshots because Figure 3 uses a deterministic, non-sensitive demo answer with the same citation format as the application.

If a Streamlit server is already running in report mode, connect to it instead:

```powershell
python scripts/generate_report_figures.py --streamlit-url http://127.0.0.1:8501
```

## Outputs

- `figures/figure_2_documents_page.png`
  - Caption: Figure 2. Document upload and management interface of the developed application.
  - In-text citation: The document-management interface shown in Figure 2 allows users to upload, index, inspect, and remove documents.
- `figures/figure_3_chat_citations.png`
  - Caption: Figure 3. Document-grounded answer and validated source citations displayed in the Chat page.
  - In-text citation: As shown in Figure 3, the assistant presents the generated answer together with the document sources cited during generation.
- `figures/figure_4_profile_page.png`
  - Caption: Figure 4. User profile configuration for adapting the level of generated explanations.
  - In-text citation: Figure 4 presents the profile-management page used to configure the user's professional role and seniority level.
- `figures/figure_5_retrieval_workflow.png`
  - Caption: Figure 5. Semantic retrieval, lexical reranking, and source-diversification workflow.
  - In-text citation: The candidate retrieval and reranking process is illustrated in Figure 5.
- `figures/figure_6_chunking_comparison.png`
  - Caption: Figure 6. Comparison of retrieval performance for different chunking configurations.
  - In-text citation: Figure 6 compares Recall@1 and Recall@5 across the evaluated chunk sizes.

## Figure 6 Data

Figure 6 is generated only when real quality experiment results exist in `evaluation/results/quality_experiments_*.json` and contain successful experiments with both Recall@1 and Recall@5. No retrieval values are invented or estimated.

## Troubleshooting

- If Playwright is missing, rerun the setup commands above.
- If Chromium is missing, run `python -m playwright install chromium`.
- If Streamlit cannot start, confirm `streamlit` is installed from `requirements.txt`.
- To generate only the programmatic diagrams and skip browser screenshots, run:

```powershell
python scripts/generate_report_figures.py --skip-screenshots
```
