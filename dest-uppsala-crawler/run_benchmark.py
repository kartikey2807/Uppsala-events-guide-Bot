import pandas as pd
import google.generativeai as genai
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.environ.get('GEMINI_API_KEY')  # Set your API key as environment variable
MODEL_NAME = 'gemini-2.5-flash'  # or 'gemini-1.5-pro', 'gemini-pro', etc.
INPUT_CSV = 'uppsala_llm_benchmark.csv'
OUTPUT_CSV = f'uppsala_benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

# Configure the Gemini API
genai.configure(api_key=API_KEY)

def evaluate_benchmark(input_file, output_file, model_name=MODEL_NAME):
    """
    Evaluate a Gemini model on the Uppsala benchmark questions.
    
    Args:
        input_file: Path to the input CSV with Category and Question columns
        output_file: Path to save results CSV
        model_name: Name of the Gemini model to use
    """
    # Read the benchmark questions - try different encodings
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(input_file, encoding='iso-8859-1')
    
    # Initialize the model
    model = genai.GenerativeModel(model_name)
    
    # Lists to store results
    responses = []
    timestamps = []
    errors = []
    
    print(f"Starting evaluation with {model_name}")
    print(f"Total questions: {len(df)}")
    print("-" * 80)
    
    # Iterate through each question
    for idx, row in df.iterrows():
        category = row['Category']
        question = row['Question']
        
        print(f"\n[{idx + 1}/{len(df)}] Category: {category}")
        print(f"Question: {question}")
        
        try:
            # Generate response
            start_time = time.time()
            response = model.generate_content(question)
            elapsed_time = time.time() - start_time
            
            # Extract the response text
            response_text = response.text if response.text else "No response generated"
            
            responses.append(response_text)
            timestamps.append(elapsed_time)
            errors.append(None)
            
            print(f"Response: {response_text[:100]}..." if len(response_text) > 100 else f"Response: {response_text}")
            print(f"Time: {elapsed_time:.2f}s")
            
            # Rate limiting - sleep briefly between requests
            time.sleep(0.5)
            
        except Exception as e:
            error_msg = str(e)
            responses.append(None)
            timestamps.append(None)
            errors.append(error_msg)
            
            print(f"Error: {error_msg}")
    
    # Add results to dataframe
    df['Response'] = responses
    df['Response_Time_Seconds'] = timestamps
    df['Error'] = errors
    df['Model'] = model_name
    df['Evaluation_Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save results
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total questions: {len(df)}")
    print(f"Successful responses: {df['Response'].notna().sum()}")
    print(f"Errors: {df['Error'].notna().sum()}")
    print(f"Average response time: {df['Response_Time_Seconds'].mean():.2f}s")
    print("=" * 80)
    
    return df

# Main execution
if __name__ == "__main__":
    # Check if API key is set
    if not API_KEY:
        print("ERROR: GOOGLE_API_KEY environment variable not set!")
        print("Please set it using: export GOOGLE_API_KEY='your-api-key-here'")
        exit(1)
    
    # Run the evaluation
    try:
        results_df = evaluate_benchmark(INPUT_CSV, OUTPUT_CSV)
        
        # Optional: Print summary statistics by category
        print("\nResults by Category:")
        print("-" * 80)
        for category in results_df['Category'].unique():
            category_df = results_df[results_df['Category'] == category]
            success_rate = (category_df['Response'].notna().sum() / len(category_df)) * 100
            avg_time = category_df['Response_Time_Seconds'].mean()
            print(f"{category}: {success_rate:.1f}% success, {avg_time:.2f}s avg time")
            
    except FileNotFoundError:
        print(f"ERROR: Input file '{INPUT_CSV}' not found!")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"ERROR: {e}")