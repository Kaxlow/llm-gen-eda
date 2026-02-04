import sys
import subprocess
import os
import re
import warnings

# --- 1. Package Installation ---
def install_packages():
    """Installs required packages if they are not present."""
    required = {'pandas', 'numpy', 'matplotlib', 'seaborn', 'wordcloud'}
    installed = {pkg.split('==')[0] for pkg in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().split('\n')}
    missing = required - installed
    
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        print("Installation complete.\n")

# Run installation immediately
install_packages()

# --- Imports after installation ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Configuration
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# --- 2. Helper Functions ---

def detect_and_mask_pii(df):
    """Detects likely PII (Email, Phone) and masks it. Returns cleaned DF and warning flag."""
    pii_found = False
    
    # Regex patterns for common PII
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'
    
    df_clean = df.copy()
    
    # Check string columns only
    for col in df_clean.select_dtypes(include=['object', 'string']):
        # check for email
        if df_clean[col].astype(str).str.contains(email_pattern).any():
            pii_found = True
            df_clean[col] = df_clean[col].astype(str).str.replace(email_pattern, '[REDACTED_EMAIL]', regex=True)
            
        # check for phone
        if df_clean[col].astype(str).str.contains(phone_pattern).any():
            pii_found = True
            df_clean[col] = df_clean[col].astype(str).str.replace(phone_pattern, '[REDACTED_PHONE]', regex=True)

    if pii_found:
        print("\n⚠️ WARNING: Potential Personally Identifiable Information (PII) detected.")
        print("Sensitive data has been masked in the output for privacy.")
        
    return df_clean

def clean_data(df, schema=None):
    """
    Performs data cleaning: renaming, whitespace trimming, date formatting, 
    placeholder replacement.
    """
    df = df.copy()
    
    # 1. Whitespace trimming (Columns and Values)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].str.strip()

    # 2. Rename columns based on context (Schema) or Heuristics
    unclear_cols = []
    if schema:
        # If schema maps names, use them
        if 'column_mapping' in schema:
            df.rename(columns=schema['column_mapping'], inplace=True)
            
    # Heuristic: Check for generic names
    for col in df.columns:
        if "Unnamed" in col or len(col) < 2:
            unclear_cols.append(col)
            # Try to give a generic meaningful name if possible, otherwise keep
            
    # 3. Replace Placeholders with NA
    placeholders = ["?", "-", "nan", "null", "missing", "N/A", ""]
    df.replace(placeholders, np.nan, inplace=True)
    
    # 4. Date Formatting
    # Attempt to convert object columns to datetime if they look like dates
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Soft check: if not majority numbers, try parsing date
                if not df[col].astype(str).str.isnumeric().all():
                    temp_dates = pd.to_datetime(df[col], errors='coerce')
                    # If valid conversion rate is high (>80%), assume it's a date col
                    if temp_dates.notna().mean() > 0.8:
                        df[col] = temp_dates.dt.strftime('%Y-%m-%d')
            except:
                pass # Not a date column

    # 5. Convert Binary (0/1) Numerical Columns to Categorical
    # Select only numerical columns first to avoid errors
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in num_cols:
        # Get unique values ignoring NaNs
        unique_vals = set(df[col].dropna().unique())
        
        # Check if the unique values are a subset of {0, 1} (e.g., {0, 1}, {0}, or {1})
        if unique_vals.issubset({0, 1}) and len(unique_vals) > 0:
            df[col] = df[col].astype('category')

    return df, unclear_cols

def generate_insights(df, num_cols, cat_cols):
    """Generates text-based insights based on statistical analysis."""
    insights = []
    
    # Insight 1: Missing Data
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.2].index.tolist()
    if high_missing:
        insights.append(f"Data Quality: Columns {high_missing} have over 20% missing data, which may impact analysis.")
    else:
        insights.append("Data Quality: The dataset is relatively complete with low missing value rates.")

    # Insight 2: Correlations (Numerical)
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper.columns if any(upper[column] > 0.85)]
        if high_corr:
            insights.append(f"Patterns: Strong multicollinearity detected in: {high_corr}. These variables move together.")
        else:
            insights.append("Patterns: Numerical variables appear relatively independent (no correlation > 0.85).")

    # Insight 3: Outliers (Numerical)
    outlier_cols = []
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            outlier_cols.append(col)
    if outlier_cols:
        insights.append(f"Anomalies: Statistical outliers detected in {len(outlier_cols)} columns (e.g., {outlier_cols[:3]}).")

    # Insight 4: Cardinality (Categorical)
    if len(cat_cols) > 0:
        unique_counts = df[cat_cols].nunique()
        high_cardinality = unique_counts[unique_counts > 50].index.tolist()
        one_val = unique_counts[unique_counts == 1].index.tolist()
        if high_cardinality:
            insights.append(f"Patterns: High variability in categorical columns: {high_cardinality[:3]}.")
        if one_val:
            insights.append(f"Anomalies: Columns {one_val} contain only one unique value and provide no information.")

    return insights

def write_summary(df, unclear_cols, insights, bias_note):
    """Writes the text summary report."""
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    with open(f"{OUTPUT_DIR}/data_summary_report.txt", "w") as f:
        f.write("DATA SUMMARY REPORT\n")
        f.write("===================\n\n")
        
        # 1. Structure
        f.write(f"Rows: {df.shape[0]}\n")
        f.write(f"Columns: {df.shape[1]}\n\n")
        
        # 2. Type Check
        if len(num_cols) == 0:
            f.write("NOTE: Dataset contains ONLY Categorical data.\n")
        elif len(cat_cols) == 0:
            f.write("NOTE: Dataset contains ONLY Numerical data.\n")
        
        # 3. Column Details
        f.write("Column Details:\n")
        f.write(f"{'Column':<25} | {'Type':<10} | {'Missing':<10} | {'Unique':<10}\n")
        f.write("-" * 65 + "\n")
        for col in df.columns:
            missing = df[col].isnull().sum()
            unique = df[col].nunique()
            dtype = str(df[col].dtype)
            f.write(f"{col:<25} | {dtype:<10} | {missing:<10} | {unique:<10}\n")
        
        if unclear_cols:
            f.write(f"\n[!] Unclear Column Meanings: {', '.join(unclear_cols)}\n")

        # 4. Descriptive Stats
        f.write("\n\nDESCRIPTIVE STATISTICS\n")
        f.write("======================\n")
        
        if len(num_cols) > 0:
            f.write("\nNumerical Statistics:\n")
            stats = df[num_cols].describe().T
            stats['IQR'] = df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)
            # Add outliers count estimate
            stats['Outliers_Count'] = stats.apply(lambda row: ((df[row.name] < (row['25%'] - 1.5*row['IQR'])) | 
                                                               (df[row.name] > (row['75%'] + 1.5*row['IQR']))).sum(), axis=1)
            f.write(stats.to_string())
        
        if len(cat_cols) > 0:
            f.write("\n\nCategorical Frequency (Top 5 per column):\n")
            for col in cat_cols:
                f.write(f"\nColumn: {col}\n")
                f.write(df[col].value_counts(normalize=True).head(5).to_string())
                f.write("\n")

        # 5. Insights
        f.write("\n\nINSIGHTS & PATTERNS\n")
        f.write("===================\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")

        # 6. Bias & Limitations
        f.write("\n\nBIAS & LIMITATIONS\n")
        f.write("==================\n")
        f.write(bias_note)

def generate_plots(df):
    """Generates and saves requested plots."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # 1. Correlation Heatmap (Numerical)
    if len(num_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
        plt.close()

    # 2. Box Plots (Separate plots for up to 3 Num variables)
    if len(num_cols) > 0:
        # Select up to 3 numerical columns
        cols_to_plot = num_cols[:3]
        
        for i, col in enumerate(cols_to_plot):
            plt.figure(figsize=(6, 6))  # Smaller size for individual plots
            sns.boxplot(y=df[col])      # Use Seaborn for a cleaner single-variable plot
            plt.title(f"Box Plot of {col}")
            plt.ylabel("Value")
            plt.tight_layout()
            
            # Save as box_plot_1.png, box_plot_2.png, etc.
            plt.savefig(f"{OUTPUT_DIR}/box_plot_{i+1}.png")
            plt.close()

    # 3. Histogram (One Num variable)
    if len(num_cols) > 0:
        target_col = num_cols[0]
        plt.figure(figsize=(8, 6))
        sns.histplot(df[target_col], kde=True)
        plt.title(f"Histogram of {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/histogram.png")
        plt.close()

    # 4. Scatterplots (Up to 3 pairs)
    if len(num_cols) >= 2:
        import itertools
        pairs = list(itertools.combinations(num_cols, 2))[:3]
        for i, (col1, col2) in enumerate(pairs):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x=col1, y=col2)
            plt.title(f"Scatterplot: {col1} vs {col2}")
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/scatter_{i+1}.png")
            plt.close()

    # 5. Word Cloud (Categorical)
    if len(cat_cols) > 0:
        # Choose the categorical column with the most unique values
        target_col = max(cat_cols, key=lambda c: df[c].nunique())
        
        # Clean the text: drop NAs, convert to string, and join
        text_data = df[target_col].dropna().astype(str)
        
        # Join and strip whitespace to see if there's actual content
        combined_text = " ".join(text_data).strip()
        
        # Further clean: extract word tokens (alphanumeric) to ensure WordCloud has tokens
        tokens = re.findall(r"\w+", combined_text)
        cleaned_text = " ".join(tokens)
        
        # Logic: Only generate if we have at least one token
        if cleaned_text:
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
            except ValueError as e:
                # WordCloud may raise ValueError if it doesn't find words after internal processing
                print(f"Skipping Word Cloud for {target_col}: {e}")
            else:
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f"Word Cloud for {target_col}")
                plt.tight_layout()
                plt.savefig(f"{OUTPUT_DIR}/word_cloud.png")
                plt.close()
        else:
            print(f"Skipping Word Cloud: No valid words found in {target_col}.")

# --- 3. Main Workflow ---

def main(csv_path, schema=None):
    print(f"Starting EDA on: {csv_path}")
    
    # Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    # PII Check
    df = detect_and_mask_pii(df)
    
    # Data Cleaning
    df, unclear_cols = clean_data(df, schema)
    
    # Identify Types
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Generate Insights
    insights = generate_insights(df, num_cols, cat_cols)
    
    # Formulate Bias Note
    bias_note = (
        "Potential Sources of Bias and Limitations:\n"
        "- Missing Values: If missing data is not random (e.g., specific groups didn't respond), results are biased.\n"
        "- Outliers: Extreme values detected in the dataset may skew mean and standard deviation calculations.\n"
        "- PII Redaction: Some text data was redacted for privacy, which may affect text analysis density.\n"
        "- Sampling: Without knowing the data source, we assume this is a representative sample, which may not be true."
    )
    
    # Generate Outputs
    print("Generating Summary Report...")
    write_summary(df, unclear_cols, insights, bias_note)
    
    print("Generating Plots...")
    generate_plots(df)
    
    print(f"\nWorkflow Complete. All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    # --- USER INPUT CONFIGURATION ---
    # Change this path to your input CSV file
    INPUT_FILE = "dataset.csv" 
    
    # Optional Schema (Example format)
    # SCHEMA = {"column_mapping": {"v1": "Age", "v2": "Income"}}
    SCHEMA = None 
    
    # Check if a file argument was passed via command line, else use default
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]

    main(INPUT_FILE, SCHEMA)