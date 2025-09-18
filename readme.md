# Mammography Extractor

## Description
The Mammography Extractor is a Python-based tool designed to process and extract data from mammography datasets. It reads input data, processes it, and saves the results to a CSV file for further analysis.

## Features
- Reads input data from CSV files.
- Processes and extracts relevant information.
- Saves the processed data to the `output/` folder.

## Prerequisites
- Python 3.7 or higher
- Virtual environment (recommended)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mammography_extractor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place your input data file in the `data/` folder.
2. Run the script:
   ```bash
   python main.py
   ```
3. The processed data will be saved to the `output/` folder as `mammography_extracted_data.csv`.

## Project Structure
```
.
├── data/                # Folder for input data
├── output/              # Folder for processed output
├── src/                 # Source code folder
├── main.py              # Main script
├── requirements.txt     # Python dependencies
├── readme.md            # Project documentation
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [Pandas](https://pandas.pydata.org/)
- [TQDM](https://tqdm.github.io/)
- [Requests](https://docs.python-requests.org/)
