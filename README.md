# Exploratory Data Analysis Workflow

This workflow takes input data as a .csv file and an optional data schema file, then performs exploratory data analysis and produces meaningful insights about the data's quality, characteristics, and inherent relationships between variables. The workflow was generated from Google Gemini 3.5.

To run the workflow:
1. Place the input .csv file and optional schema file in the `data` folder
2. Open a terminal session and navigate to the project's src directory `\llm-gen-eda\src`
3. Run the following command, replacing the parameters in parentheses with the input file paths:
    python3 eda_workflow.py [..\data\input.csv] [..\data\optional_schema.json]
4. The output is available in the `\llm-gen-eda\output` directory