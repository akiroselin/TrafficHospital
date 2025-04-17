# LangChain Data Analysis Agent

A powerful data analysis agent built with LangChain that can load, analyze, and visualize data using natural language commands.

## Overview

This project implements a data analysis agent using the LangChain framework. The agent can:

- Load data from CSV and Excel files
- Generate sample datasets for testing
- Provide detailed information about datasets
- Run queries on datasets using pandas query syntax
- Create various visualizations (histograms, scatter plots, bar charts, etc.)
- Perform common analyses (summary statistics, correlation analysis, outlier detection, etc.)
- Answer questions about the data in natural language

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd langchain-data-analysis-agent
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Ollama configuration:
   ```
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3
   ```

## Usage

Run the main script to start the interactive data analysis agent:

```
python main.py
```

The agent will automatically generate a sample sales dataset to help you get started. You can then interact with the agent using natural language commands.

### Example Commands

- "Show me a summary of the sales data"
- "Create a bar chart of revenue by region"
- "Which product has the highest profit margin?"
- "Generate a weather dataset with 200 rows"
- "Load my data from 'path/to/my/data.csv'"
- "Show me the correlation between revenue and units sold"
- "Find outliers in the profit column"
- "Create a scatter plot of units sold vs revenue"

## Available Tools

The agent has access to the following tools:

### Data Loading

- **load_csv**: Load a CSV file into memory
  ```
  Example: "Load the data from 'sales_data.csv'"
  ```

- **load_excel**: Load an Excel file into memory
  ```
  Example: "Import the Excel file at 'quarterly_report.xlsx'"
  ```

- **generate_sample_data**: Generate a sample dataset for testing
  ```
  Example: "Create a sample stock market dataset with 500 rows"
  ```

### Data Exploration

- **list_datasets**: List all datasets currently loaded in memory
  ```
  Example: "What datasets do I have loaded?"
  ```

- **get_dataset_info**: Get detailed information about a dataset
  ```
  Example: "Tell me about the sales_data dataset"
  ```

- **describe_column**: Get detailed statistics about a specific column
  ```
  Example: "Describe the revenue column in the sales_data dataset"
  ```

### Data Analysis

- **query_data**: Run a query on a dataset using pandas query syntax
  ```
  Example: "Show me records where revenue > 5000"
  ```

- **run_analysis**: Run a predefined analysis on a dataset
  ```
  Example: "Run a correlation analysis on the sales_data"
  ```

### Data Visualization

- **create_visualization**: Create a static visualization of the data
  ```
  Example: "Create a histogram of the temperature column in the weather_data"
  ```

- **create_interactive_visualization**: Create an interactive visualization with hover information
  ```
  Example: "Create an interactive scatter plot of price vs sales with product as color"
  ```

## Advanced Usage

### Custom Analyses

You can ask the agent to perform custom analyses by combining multiple tools:

```
"First, show me the distribution of sales by region, then analyze the correlation between units sold and profit, and finally identify any outliers in the revenue data."
```

### Combining Visualizations with Analysis

```
"Create a scatter plot of revenue vs profit, and highlight any strong correlations you find."
```

### Asking Complex Questions

```
"Which sales channel is most effective in the East region, and how does its performance compare to other regions?"
```

## Extending the Agent

You can extend the agent's capabilities by adding new tools to the `data_tools.py` file and registering them in the `create_data_analysis_agent` function in `main.py`.

## License

[MIT License](LICENSE)
