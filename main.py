# Install required packages (run this cell first in Colab)
# !pip install pandas tqdm requests

import pandas as pd
import json
import time
import requests
from typing import Dict, Any
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MammographyExtractor:
    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "llama3.1:8b"):
        """
        Initialize the mammography data extractor using local Ollama
        
        Args:
            ollama_url: URL where Ollama is running (default: http://localhost:11434)
            model_name: Name of the Ollama model to use (e.g., 'llama3.1:8b', 'mistral:7b', 'qwen2.5:7b')
        """
        self.ollama_url = ollama_url.rstrip('/')
        self.model_name = model_name
        self.api_endpoint = f"{self.ollama_url}/api/generate"
        
        # Test connection to Ollama
        try:
            self._test_connection()
            logger.info(f"✅ Successfully connected to Ollama at {self.ollama_url}")
            logger.info(f"✅ Using model: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Error connecting to Ollama: {e}")
            raise Exception(f"Please ensure Ollama is running at {self.ollama_url} with model {self.model_name}")
        
        # Define the extraction prompt
        self.extraction_prompt = """
You are a medical AI assistant specialized in analyzing mammography reports. Extract the following information from the mammography report text for BOTH right and left breasts separately. If information is not mentioned or not applicable, use "not mentioned" or "N/A".

Extract the following data in JSON format:
{
  "right_breast": {
    "birads": "BI-RADS category (1, 2, 3, 4, 5, or 6)",
    "classification": "primary classification (negative, benign, probably benign, suspicious, malignant)",
    "calcification_present": "yes or no, and type if present (benign, dystrophic, scattered, clustered, microcalcifications, etc.)",
    "previous_surgery": "type of previous surgery (mastectomy, BCS, lumpectomy, none, or not mentioned)",
    "normal_status": "normal or abnormal findings",
    "benign_vs_malignant": "benign, malignant, suspicious, or normal",
    "density": "heterogenous, fatty, extremely dense, scattered fibroglandular, or not mentioned",
    "cyst_present": "yes or no (yes if any cysts mentioned, no if not mentioned or explicitly absent)"
  },
  "left_breast": {
    "birads": "BI-RADS category (1, 2, 3, 4, 5, or 6)",
    "classification": "primary classification (negative, benign, probably benign, suspicious, malignant)",
    "calcification_present": "yes or no, and type if present (benign, dystrophic, scattered, clustered, microcalcifications, etc.)",
    "previous_surgery": "type of previous surgery (mastectomy, BCS, lumpectomy, none, or not mentioned)",
    "normal_status": "normal or abnormal findings", 
    "benign_vs_malignant": "benign, malignant, suspicious, or normal",
    "density": "heterogenous, fatty, extremely dense, scattered fibroglandular, or not mentioned",
    "cyst_present": "yes or no (yes if any cysts mentioned, no if not mentioned or explicitly absent)"
  }
}

Important notes:
- If the report mentions "bilateral" findings, apply to both breasts
- Look for terms like "heterogeneously dense", "extremely dense", "predominantly fat", "scattered fibroglandular"
- Previous surgery terms: "status post mastectomy", "status post BCS", "breast conserving surgery"
- BI-RADS categories: 1=negative, 2=benign, 3=probably benign, 4=suspicious, 5=highly suspicious, 6=malignant
- For cyst detection, look for terms: "cyst", "cysts", "cystic", "complicated cyst", "simple cyst"
- For calcification detection, look for: "calcification", "calcifications", "microcalcifications", "clustered", "scattered", "benign calcifications", "dystrophic calcifications"
- Return only valid JSON, no additional text

Extract data from this mammography report:

"""

    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": "Test connection",
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama server returned status code {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {self.ollama_url}. Please ensure Ollama is running.")
        except requests.exceptions.Timeout:
            raise Exception(f"Timeout connecting to Ollama. Please check if the model {self.model_name} is available.")

    def extract_from_text(self, report_text: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Extract mammography data from a single report text using Ollama
        
        Args:
            report_text: The mammography report text
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing extracted data for both breasts
        """
        full_prompt = self.extraction_prompt + report_text
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9,
                            "max_tokens": 1000
                        }
                    },
                    timeout=60
                )
                
                if response.status_code != 200:
                    logger.warning(f"Ollama API returned status code {response.status_code}, attempt {attempt + 1}/{max_retries}")
                    continue
                
                result = response.json()
                result_text = result.get('response', '').strip()
                
                # Clean up the response (remove any markdown formatting)
                if result_text.startswith("```json"):
                    result_text = result_text[7:-3]
                elif result_text.startswith("```"):
                    result_text = result_text[3:-3]
                
                # Try to find JSON in the response
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_text = result_text[json_start:json_end]
                    extracted_data = json.loads(json_text)
                    return extracted_data
                else:
                    logger.warning(f"No valid JSON found in response, attempt {attempt + 1}/{max_retries}")
                    continue
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to parse JSON after {max_retries} attempts")
                    logger.error(f"Last response was: {result_text[:500]}...")
                continue
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            except Exception as e:
                logger.warning(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
                continue
        
        logger.error("All extraction attempts failed, returning empty structure")
        return self._get_empty_structure()

    def _get_empty_structure(self) -> Dict[str, Any]:
        """Return empty structure when extraction fails"""
        return {
            "right_breast": {
                "birads": "not mentioned",
                "classification": "not mentioned", 
                "calcification_present": "not mentioned",
                "previous_surgery": "not mentioned",
                "normal_status": "not mentioned",
                "benign_vs_malignant": "not mentioned",
                "density": "not mentioned",
                "cyst_present": "not mentioned"
            },
            "left_breast": {
                "birads": "not mentioned",
                "classification": "not mentioned",
                "calcification_present": "not mentioned",
                "previous_surgery": "not mentioned", 
                "normal_status": "not mentioned",
                "benign_vs_malignant": "not mentioned",
                "density": "not mentioned",
                "cyst_present": "not mentioned"
            }
        }

    def process_dataframe(self, df: pd.DataFrame, text_column: str = '_text', 
                         batch_size: int = 5, delay: float = 2.0) -> pd.DataFrame:
        """
        Process a DataFrame containing mammography reports
        
        Args:
            df: DataFrame with mammography reports
            text_column: Name of column containing report text
            batch_size: Number of reports to process before a delay (smaller for local processing)
            delay: Delay in seconds between batches
            
        Returns:
            DataFrame with extracted features added
        """
        results = []
        total_rows = len(df)
        
        print(f"🔄 Starting processing of {total_rows} mammography reports using {self.model_name}...")
        
        # Create progress bar
        with tqdm(total=total_rows, desc="Processing Reports", unit="report") as pbar:
            for idx, row in df.iterrows():
                # Extract data from the report text
                extracted_data = self.extract_from_text(row[text_column])
                
                # Create a flat structure for DataFrame columns
                flat_data = {
                    'right_birads': extracted_data['right_breast']['birads'],
                    'right_classification': extracted_data['right_breast']['classification'],
                    'right_calcification_present': extracted_data['right_breast']['calcification_present'],
                    'right_previous_surgery': extracted_data['right_breast']['previous_surgery'],
                    'right_normal': extracted_data['right_breast']['normal_status'],
                    'right_benign_vs_malignant': extracted_data['right_breast']['benign_vs_malignant'],
                    'right_density': extracted_data['right_breast']['density'],
                    'right_cyst_present': extracted_data['right_breast']['cyst_present'],
                    'left_birads': extracted_data['left_breast']['birads'],
                    'left_classification': extracted_data['left_breast']['classification'],
                    'left_calcification_present': extracted_data['left_breast']['calcification_present'],
                    'left_previous_surgery': extracted_data['left_breast']['previous_surgery'],
                    'left_normal': extracted_data['left_breast']['normal_status'],
                    'left_benign_vs_malignant': extracted_data['left_breast']['benign_vs_malignant'],
                    'left_density': extracted_data['left_breast']['density'],
                    'left_cyst_present': extracted_data['left_breast']['cyst_present']
                }
                
                results.append(flat_data)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Current': f"{idx + 1}/{total_rows}",
                    'Batch': f"{(idx + 1) % batch_size}/{batch_size}"
                })
                
                # Add delay between batches to manage local processing load
                if (idx + 1) % batch_size == 0:
                    pbar.set_description(f"Processing Reports - Resting {delay}s")
                    time.sleep(delay)
                    pbar.set_description("Processing Reports")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Combine with original DataFrame
        final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        print("✅ Processing completed successfully!")
        return final_df

    def display_summary(self, df: pd.DataFrame):
        """Display a summary of the extracted data"""
        print("\n" + "="*50)
        print("📊 EXTRACTION SUMMARY")
        print("="*50)
        
        # Right breast summary
        print("\n🟦 RIGHT BREAST SUMMARY:")
        print(f"BI-RADS Distribution: {df['right_birads'].value_counts().to_dict()}")
        print(f"Density Distribution: {df['right_density'].value_counts().to_dict()}")
        print(f"Surgery History: {df['right_previous_surgery'].value_counts().to_dict()}")
        print(f"Calcification Present: {df['right_calcification_present'].value_counts().to_dict()}")
        print(f"Cyst Present: {df['right_cyst_present'].value_counts().to_dict()}")
        
        # Left breast summary  
        print("\n🟩 LEFT BREAST SUMMARY:")
        print(f"BI-RADS Distribution: {df['left_birads'].value_counts().to_dict()}")
        print(f"Density Distribution: {df['left_density'].value_counts().to_dict()}")
        print(f"Surgery History: {df['left_previous_surgery'].value_counts().to_dict()}")
        print(f"Calcification Present: {df['left_calcification_present'].value_counts().to_dict()}")
        print(f"Cyst Present: {df['left_cyst_present'].value_counts().to_dict()}")
        
        print("\n" + "="*50)

# Quick setup and demo function
def run_mammography_extraction(df=None, text_column='_text', ollama_url="http://localhost:11434", model_name="llama3.1:8b"):
    """
    Main function to run mammography extraction with Ollama
    
    Args:
        df: DataFrame with mammography reports (optional - will create demo if None)
        text_column: Name of column containing report text
        ollama_url: URL where Ollama is running
        model_name: Name of the Ollama model to use
    """
    
    # Initialize the extractor
    print(f"🚀 Initializing Mammography Extractor with Ollama...")
    print(f"🔗 Connecting to: {ollama_url}")
    print(f"🤖 Using model: {model_name}")
    
    try:
        extractor = MammographyExtractor(ollama_url=ollama_url, model_name=model_name)
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        return None
    
    # Use provided DataFrame or create sample data for demo
    if df is None:
        print("📝 No DataFrame provided, creating sample data for demonstration...")
        sample_data = {
            '_text': [
                """bilateral diagnostic digital mammogram clinical history: 61 year-old female, had history of left breast cancer status post bcs. findings: the breast tissues are heterogeneously dense. postoperative changes at the left axilla and left subareolar aspect with coarse calcification suggestive of fat necrosis. scattered bilateral benign appearing calcifications. cysts at left breast. impression: postoperative changes in the left breast and left axilla. bilateral benign looking calcifications. birads 2- benign finding.""",
                """right diagnostic digital mammogram clinical indication: a 68-year-old female patient. left breast cancer in 2003, status post left mastectomy. findings: the breast tissues are extremely dense. scattered cysts of variable complexity. dystrophic calcifications present. impression: the right breast demonstrates dense parenchyma with marked fibrocystic change. birads 3 : probably benign findings""",
                """bilateral screening digital mammogram clinical history: 53-year-old lady. for screening mammography. findings: the breast tissue are heterogeneously dense. scattered benign looking calcifications. there are few small cysts bilateral mainly in the left breast. there is no evidence of suspicious clustered microcalcifications. impression: no mammographic evidence of malignancy. birads 2- benign findings."""
            ]
        }
        df = pd.DataFrame(sample_data)
    
    # Process the DataFrame
    print(f"\n🔬 Processing {len(df)} mammography reports...")
    processed_df = extractor.process_dataframe(df, text_column, batch_size=3, delay=1.0)
    
    # Display summary
    extractor.display_summary(processed_df)
    
    # Save to CSV
    filename = "./output/mammography_extracted_data.csv"
    processed_df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to {filename}")
    
    # Display first few rows
    print(f"\n📋 SAMPLE OF EXTRACTED DATA:")
    print("="*80)
    
    # Show only the extracted columns for better readability
    extracted_cols = [col for col in processed_df.columns if col.startswith(('right_', 'left_'))]
    print(processed_df[extracted_cols].head())
    
    return processed_df

# Load and process your data
print("📂 Loading data from /data/datanew.csv...")
try:
    df = pd.read_csv('/data/datanew.csv')
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Check if _text column exists
    if '_text' in df.columns:
        print(f"✅ '_text' column found with {df['_text'].notna().sum()} non-null entries")
        
        # Show sample of first few characters from _text column
        print(f"📝 Sample text preview:")
        print("-" * 50)
        for i in range(min(3, len(df))):
            sample_text = str(df['_text'].iloc[i])[:200] + "..." if len(str(df['_text'].iloc[i])) > 200 else str(df['_text'].iloc[i])
            print(f"Row {i+1}: {sample_text}")
            print()
        
        # Run the extraction with Ollama
        print("🚀 Starting mammography data extraction with Ollama...")
        print("⚠️  Make sure Ollama is running locally with your chosen model!")
        
        # You can change these parameters:
        OLLAMA_URL = "http://localhost:11434"  # Change if Ollama runs on different port
        MODEL_NAME = "llama3.1:8b"             # Change to your preferred model
        
        processed_df = run_mammography_extraction(
            df, 
            text_column='_text',
            ollama_url=OLLAMA_URL,
            model_name=MODEL_NAME
        )
        
    else:
        print("❌ Error: '_text' column not found in the dataset")
        print(f"Available columns: {list(df.columns)}")
        
except FileNotFoundError:
    print("❌ Error: File '/data/datanew.csv' not found")
    print("Please ensure your CSV file is uploaded to Colab at this path")
except Exception as e:
    print(f"❌ Error loading data: {e}")

# Keep the demo function available for testing
def run_demo():
    """Run with sample data for testing"""
    return run_mammography_extraction()

# Instructions for local Ollama users
print("""
🔧 OLLAMA SETUP INSTRUCTIONS:

1. Install Ollama on your local machine:
   - Visit: https://ollama.ai
   - Download and install Ollama
   
2. Pull a language model (run in terminal):
   ollama pull llama3.1:8b
   # or try: mistral:7b, qwen2.5:7b, llama3.2:3b
   
3. Start Ollama server (if not auto-started):
   ollama serve
   
4. Install required Python packages:
   !pip install pandas tqdm requests

5. If running in Colab, you'll need to tunnel to your local Ollama:
   - Use ngrok or similar tool to expose localhost:11434
   - Update OLLAMA_URL in the code above

6. Popular models for medical text:
   - llama3.1:8b (recommended, good balance)
   - mistral:7b (faster, lighter)  
   - qwen2.5:7b (good for structured output)
   - llama3.2:3b (very fast, less accurate)

7. The extraction will process reports locally on your machine!
""")

# Example usage
if __name__ == "__main__":
    # Uncomment to run demo
    # processed_df = run_demo()
    pass