# Mammography Extractor Manual

## Overview
This guide walks you through setting up the environment and running the mammography extraction pipeline that lives in `main.py`. The script reads mammography reports from a CSV file, sends each report to a local Ollama model, and writes the structured results to `output/mammography_extracted_data.csv`.

## Prerequisites
- Python 3.9 or newer with `pip`
- (Recommended) Git for cloning the repository
- A CSV file containing mammography reports with a `_text` column
- [Ollama](https://ollama.ai) installed locally with at least one model pulled (e.g. `llama3.1:8b`)
- Internet access is **not** required once dependencies and Ollama models are installed

## 1. Obtain the Project
```bash
# Clone with Git
git clone <repository-url>
cd mammography_extractor

# Or download and unzip the project, then open a terminal here
```

## 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

## 3. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> Note: `logging` is part of Python's standard library. If `pip` warns about it you can ignore the message.

## 4. Prepare Ollama
1. Install Ollama from https://ollama.ai and launch it.
2. Pull a compatible model (run in a separate terminal):
   ```bash
   ollama pull llama3.1:8b
   # alternatives: mistral:7b, qwen2.5:7b, llama3.2:3b
   ```
3. Ensure the Ollama server is listening on `http://localhost:11434`. If it is not already running, start it:
   ```bash
   ollama serve
   ```
4. If Ollama runs on a different host or port, update `OLLAMA_URL` and `MODEL_NAME` in `main.py` before executing the script.

## 5. Add Your Input Data
- Place your CSV file in the `data/` directory.
- The default script expects the file at `data/datanew.csv` and requires a `_text` column containing the raw report text.
- Adjust the path in `main.py` if you want to use a different filename or location.

## 6. Run the Extractor
```bash
python main.py
```
During execution the script will:
- Verify that the Ollama endpoint is reachable.
- Load the dataset from `data/datanew.csv`.
- Display basic information about the dataset and sample report text.
- Process the reports in batches, sending them to the selected Ollama model.
- Save the structured output to `output/mammography_extracted_data.csv`.

## 7. Review Results
- The CSV in `output/mammography_extracted_data.csv` contains one row per report along with extracted fields such as BI-RADS category, calcification status, density, and cyst presence for each breast.
- Inspect the file with your preferred spreadsheet viewer or `pandas` to validate the output.

## Optional: Run the Built-in Demo
If you want to test the pipeline without your own data, you can enable the demo function inside `main.py`:
1. Open `main.py` and locate the `if __name__ == "__main__":` block at the bottom.
2. Uncomment the `processed_df = run_demo()` line.
3. Run `python main.py` to generate sample output in the `output/` directory.

## Troubleshooting
- **Connection errors to Ollama**: Confirm `ollama serve` is running and that the `OLLAMA_URL` in `main.py` matches the correct host and port.
- **Missing `_text` column**: Ensure the input CSV has a `_text` column or update the code to use the correct column name.
- **File not found**: Verify the CSV path and that the `data/` directory exists.
- **Permission issues on Windows**: Run your terminal as Administrator if you see access denied errors when installing packages or writing output files.
