# Evidence Base Classifier

This project provides a Streamlit application for automatically reviewing academic articles in PDF format. It extracts the text from PDFs, sends the text to a large language model, and records the analysis results.

## Features

- Upload one or multiple PDF files directly in the web interface.
- Text is extracted using `pdfplumber`; if insufficient text is found, OCR is attempted with `pytesseract`.
- Integration with **OpenAI** (`o3`) and **Anthropic** (`claude-opus-4`) language models via the `LLMClient` class.
- Structured JSON output is validated against a schema to ensure fields such as `article_title`, `inclusion_decision`, and `category` are always present.
- Results are managed by `ResultsManager`, which can export successful analyses and error logs to CSV or text files in the `output/` and `logs/` directories.
- Optional configuration file stored at `~/.evidence_base_classifier/config.yaml` if you choose to persist API keys and settings.

## Installation

1. Install Python 3.12 or later.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

`pytesseract` relies on the Tesseract OCR engine. Make sure it is installed on your system.

## Running the Application

Execute the Streamlit app from the repository root:

```bash
streamlit run src/app.py
```

In the sidebar you can enter your OpenAI or Anthropic API keys, choose the language model, and upload PDF files for processing. If "Save Configuration" is enabled, the settings are written to `~/.evidence_base_classifier/config.yaml`.

## Repository Structure

- `src/app.py` – Streamlit user interface and overall workflow for processing PDFs.
- `src/pdf_processor.py` – Extracts text from PDFs with OCR fallback and content cleaning.
- `src/llm_client.py` – Handles calls to OpenAI and Anthropic APIs, including streaming support and result validation.
- `src/results_manager.py` – Stores analysis results, summarizes statistics, and exports CSV or text logs.
- `requirements.txt` – Python dependencies.

## Output

Processed results are written to the `output/` directory with a timestamped CSV file name. Any errors encountered during processing are saved to CSV and text logs in `output/` and `logs/`. Example fields include:

- `article_title`
- `inclusion_exclusion_decision`
- `category`
- `detailed_reasoning_for_decision`
- `source_file`

## Notes

- The application estimates token counts to warn about extremely large documents.
- Anthropic's streaming API is used for long documents with `claude-opus-4`.
- Excluded papers automatically receive a category of `N/A` during validation.

