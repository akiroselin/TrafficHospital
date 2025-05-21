"""
Data analysis tools for LangChain agent.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from typing import List, Dict, Any, Optional, Union
from langchain.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field 
from io import StringIO


# Global variable to store loaded datasets
DATASETS = {}

# --- Pydantic Models for Chart.js Structure ---
class ChartJsDataset(BaseModel):
    label: str
    data: List[Any] 
    borderColor: Optional[str] = None
    backgroundColor: Optional[str] = None
    tension: Optional[float] = None
    fill: Optional[bool] = None

class ChartJsData(BaseModel):
    labels: List[str]
    datasets: List[ChartJsDataset]

class ChartJsScaleTitle(BaseModel):
    display: bool
    text: str

class ChartJsScale(BaseModel):
    type: str
    beginAtZero: Optional[bool] = None
    title: Optional[ChartJsScaleTitle] = None

class ChartJsScales(BaseModel):
    x: Optional[ChartJsScale] = None
    y: Optional[ChartJsScale] = None

class ChartJsLegend(BaseModel):
    display: bool
    position: Optional[str] = None

class ChartJsTitle(BaseModel):
    display: bool
    text: str

class ChartJsPlugins(BaseModel):
    legend: Optional[ChartJsLegend] = None
    title: Optional[ChartJsTitle] = None

class ChartJsOptions(BaseModel):
    responsive: bool
    maintainAspectRatio: Optional[bool] = None
    plugins: Optional[ChartJsPlugins] = None
    scales: Optional[ChartJsScales] = None

class ChartJsPayload(BaseModel):
    type: str
    data: ChartJsData
    options: ChartJsOptions
# --- End Pydantic Models ---


class DataFrameInfo(BaseModel):
    """Information about a pandas DataFrame."""
    shape: tuple = Field(..., description="Shape of the DataFrame (rows, columns)")
    columns: List[str] = Field(..., description="List of column names")
    dtypes: Dict[str, str] = Field(..., description="Data types of each column")
    head: str = Field(..., description="String representation of the first few rows")
    description: str = Field(..., description="Statistical description of numerical columns")
    missing_values: Dict[str, int] = Field(..., description="Count of missing values per column")

@tool
def load_csv(file_path: str) -> str:
    """
    Load a CSV file into a pandas DataFrame and store it in memory.
    The dataset will be named after the CSV file (without extension).
    Args:
        file_path: Path to the CSV file to load.
    Returns:
        A message indicating success or failure, including dataset name and shape.
    """
    try:
        if not os.path.exists(file_path): return f"Error: File '{file_path}' does not exist."
        df = pd.read_csv(file_path)
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        DATASETS[dataset_name] = df
        return f"Successfully loaded '{file_path}' as dataset '{dataset_name}' with shape {df.shape}"
    except Exception as e: return f"Error loading CSV file: {str(e)}"

@tool
def load_excel(file_path: str, sheet_name: Optional[str] = None) -> str:
    """
    Load a specific sheet from an Excel file into a pandas DataFrame and store it in memory.
    If sheet_name is None and the Excel file has only one sheet, that sheet is loaded.
    If sheet_name is None and multiple sheets exist, an error message listing available sheets is returned.
    The dataset will be named after the Excel file (without extension).
    Args:
        file_path: Path to the Excel file to load.
        sheet_name: Optional name of the sheet to load. If None, attempts to load the first sheet or prompts if multiple exist.
    Returns:
        A message indicating success, failure, or a prompt to specify a sheet name.
    """
    try:
        if not os.path.exists(file_path): return f"Error: File '{file_path}' does not exist."
        xls = pd.ExcelFile(file_path)
        available_sheets = xls.sheet_names
        if not available_sheets: return f"Error: No sheets found in Excel file '{file_path}'."
        
        df = None
        loaded_sheet_name = ""

        if sheet_name is not None: 
            if sheet_name in available_sheets:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                loaded_sheet_name = sheet_name
            else: return (f"Error: Sheet '{sheet_name}' not found in '{file_path}'. "
                        f"Available sheets are: {', '.join(available_sheets)}.")
        else: 
            if len(available_sheets) == 1:
                df = pd.read_excel(xls, sheet_name=available_sheets[0])
                loaded_sheet_name = available_sheets[0]
            else: 
                return (f"Excel file '{file_path}' contains multiple sheets: {', '.join(available_sheets)}. "
                        "Please specify which sheet to load using the 'sheet_name' parameter.")
        
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        DATASETS[dataset_name] = df
        
        msg = f"Successfully loaded sheet '{loaded_sheet_name}' from '{file_path}' as dataset '{dataset_name}' with shape {df.shape}."
        return msg
    except Exception as e: return f"Error processing Excel file '{file_path}': {str(e)}"

@tool
def list_datasets() -> str:
    """
    List all datasets currently loaded in memory, along with their shapes.
    Returns:
        A string listing available datasets or a message if no datasets are loaded.
    """
    if not DATASETS: return "No datasets currently loaded."
    return "Available datasets:\n" + "\n".join([f"- {name}: shape {df.shape}" for name, df in DATASETS.items()])

@tool
def get_dataset_info(dataset_name: str) -> str:
    """
    Get detailed information about a specified dataset loaded in memory.
    This includes shape, column names, data types, a preview (head),
    summary statistics, and missing value counts for each column.
    Args:
        dataset_name: The name of the dataset to get information about.
    Returns:
        A string with detailed information or an error if the dataset is not found.
    """
    if dataset_name not in DATASETS: return f"Error: Dataset '{dataset_name}' not found."
    df = DATASETS[dataset_name]
    info = DataFrameInfo(shape=df.shape, columns=df.columns.tolist(), dtypes={c: str(t) for c,t in df.dtypes.items()}, head=df.head().to_string(), description=df.describe().to_string(), missing_values={c:int(df[c].isna().sum()) for c in df.columns})
    result = f"Dataset: {dataset_name}\nShape: {info.shape[0]} rows × {info.shape[1]} columns\n\nColumns:\n"
    for col, dtype in info.dtypes.items():
        missing = info.missing_values[col]
        missing_pct = (missing / info.shape[0] * 100) if info.shape[0] > 0 else 0
        result += f"- {col} ({dtype}): {missing} missing values ({missing_pct:.1f}%)\n"
    result += f"\nPreview:\n{info.head}\n\nSummary Statistics:\n{info.description}\n"
    return result

@tool
def query_data(dataset_name: str, query: str) -> str:
    """
    Run a query on a loaded dataset using pandas .query() syntax.
    Args:
        dataset_name: The name of the dataset to query.
        query: The query string (e.g., "column_name > 10 and other_column == 'value'").
    Returns:
        A string representation of the query results (first 10 rows) or an error message.
    """
    if dataset_name not in DATASETS: return f"Error: Dataset '{dataset_name}' not found."
    try:
        df = DATASETS[dataset_name]
        res_df = df.query(query)
        return f"Query returned {len(res_df)} rows. Preview (first 10 rows):\n{res_df.head(10).to_string()}"
    except Exception as e: return f"Error executing query: {str(e)}"

@tool
def describe_column(dataset_name: str, column_name: str) -> str:
    """
    Get detailed statistics about a specific column in a loaded dataset.
    Includes data type, non-null count, missing values, and type-specific stats
    (min, max, mean, median, std, quartiles for numeric; unique counts, top values for string).
    Args:
        dataset_name: The name of the dataset.
        column_name: The name of the column to describe.
    Returns:
        A string with detailed statistics or an error message.
    """
    if dataset_name not in DATASETS: return f"Error: Dataset '{dataset_name}' not found."
    df = DATASETS[dataset_name]
    if column_name not in df.columns: return f"Error: Column '{column_name}' not found in dataset '{dataset_name}'. Available columns: {df.columns.tolist()}"
    try:
        s = df[column_name]
        desc = f"Column: {column_name}\nType: {s.dtype}\nNonNull: {s.count()}/{len(s)}\nMissing: {s.isna().sum()} ({s.isna().mean():.1%})\n"
        if pd.api.types.is_numeric_dtype(s): desc += f"Stats: Min={s.min()}, Max={s.max()}, Mean={s.mean():.2f}, Median={s.median():.2f}, StdDev={s.std():.2f}\nQuartiles:\n{s.quantile([0.25, 0.5, 0.75]).to_string()}\n"
        elif pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s): 
            desc += f"Unique values: {s.nunique()}\nTop 10 values:\n{s.value_counts().head(10).to_string()}\n"
            if s.nunique() > 10: desc += f"Note: Only showing top 10 out of {s.nunique()} unique values.\n"
        else: desc += "No specific summary statistics for this data type."
        return desc
    except Exception as e: return f"Error describing column '{column_name}': {str(e)}"

@tool
def create_visualization( # Static Matplotlib/Seaborn plots
    dataset_name: str, plot_type: str, x: Optional[str] = None, y: Optional[str] = None,
    hue: Optional[str] = None, filename: Optional[str] = None
) -> str: 
    """
    Create a static (PNG) visualization using Matplotlib/Seaborn and save it to a file.
    Supported plot_types: histogram, scatter, bar, box, line, heatmap, pair.
    Args:
        dataset_name: Name of the dataset to visualize.
        plot_type: Type of plot to create.
        x: Column name for x-axis (behavior varies by plot_type).
        y: Column name for y-axis (behavior varies by plot_type).
        hue: Column name for color grouping.
        filename: Optional name for the output PNG file (without extension).
    Returns:
        A message indicating success or failure and the path to the saved plot.
    """
    if dataset_name not in DATASETS: return f"Error: Dataset '{dataset_name}' not found."
    df = DATASETS[dataset_name]
    if x and x not in df.columns: return f"Error: Column '{x}' for x-axis not found."
    if y and y not in df.columns: return f"Error: Column '{y}' for y-axis not found."
    if hue and hue not in df.columns: return f"Error: Column '{hue}' for hue not found."

    if not filename: filename = f"{dataset_name}_{plot_type}_static"
    filename = os.path.splitext(filename)[0]
    output_path = f"{filename}.png"
    
    plt.figure(figsize=(10, 6))
    try:
        if plot_type == "histogram":
            if not x: return "Error: X-axis column is required for histogram."
            sns.histplot(data=df, x=x, hue=hue); plt.title(f"Histogram of {x}")
        elif plot_type == "scatter":
            if not x or not y: return "Error: X and Y columns are required for scatter plot."
            sns.scatterplot(data=df, x=x, y=y, hue=hue); plt.title(f"Scatter plot of {y} vs {x}")
        elif plot_type == "bar": 
            if not x or not y: return "Error: X and Y columns are required for bar plot."
            sns.barplot(data=df, x=x, y=y, hue=hue); plt.title(f"Bar plot of {y} by {x}")
        elif plot_type == "box":
            if not y: return "Error: Y column is required for box plot." 
            sns.boxplot(data=df, x=x, y=y, hue=hue); plt.title(f"Box plot of {y}" + (f" by {x}" if x else ""))
        elif plot_type == "line":
            if not x or not y: return "Error: X and Y columns are required for line plot."
            sns.lineplot(data=df, x=x, y=y, hue=hue); plt.title(f"Line plot of {y} vs {x}")
        elif plot_type == "heatmap":
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.empty: return "Error: No numeric columns for heatmap."
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f"); plt.title("Correlation Heatmap")
        elif plot_type == "pair":
            sns.pairplot(df.select_dtypes(include=np.number), hue=hue); plt.suptitle(f"Pair Plot of Numeric Columns", y=1.02)
        else: return f"Error: Static plot type '{plot_type}' not supported."
        
        plt.tight_layout(); plt.savefig(output_path);
        return f"Successfully created static {plot_type} plot: '{output_path}'."
    except Exception as e: return f"Error creating static visualization: {str(e)}"
    finally: plt.close()

@tool
def create_interactive_visualization(
    dataset_name: str, 
    plot_type: str, 
    x: Optional[str] = None, 
    y: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
    facet: Optional[str] = None,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    DEBUG VERSION: Returns FORCED DEBUG Chart.js data for line plots.
    Saves actual Plotly HTML/PNG files for line plots if possible.
    Args:
        dataset_name: Name of the dataset.
        plot_type: Type of plot.
        x: Column for x-axis.
        y: Column for y-axis.
        ... (other params)
    Returns:
        A dictionary with 'textSummary' and 'chartData'.
    """
    print(f"[DEBUG data_tools.py] create_interactive_visualization (DEBUG MODE) called with:")
    print(f"  dataset_name: {dataset_name}, plot_type: {plot_type}, x: {x}, y: {y}")

    if dataset_name not in DATASETS:
        return {"textSummary": f"Error: Dataset '{dataset_name}' not found.", "chartData": None}
    df = DATASETS[dataset_name] 

    if x and x not in df.columns:
        return {"textSummary": f"Error: Column '{x}' (for x-axis) not found. Available: {df.columns.tolist()}", "chartData": None}
    if y and y not in df.columns:
        return {"textSummary": f"Error: Column '{y}' (for y-axis) not found. Available: {df.columns.tolist()}", "chartData": None}

    summary_text = f"Debug: Tool called for {plot_type}."
    fig = None 
    output_path_html, output_path_png = None, None 

    if plot_type == "line" and x and y:
        try:
            if not filename: temp_filename = f"{dataset_name}_{plot_type}_interactive_debug"
            else: temp_filename = filename
            temp_filename = os.path.splitext(temp_filename)[0]
            
            df_plot = df.copy()
            # Attempt to convert x to datetime if it looks like a time/date string column
            if pd.api.types.is_string_dtype(df_plot[x]) or pd.api.types.is_object_dtype(df_plot[x]):
                try:
                    df_plot[x] = pd.to_datetime(df_plot[x], errors='coerce')
                    if df_plot[x].isnull().all(): df_plot[x] = df[x] # revert if all NaT
                except: pass 

            fig = px.line(df_plot, x=x, y=y, color=color, facet_col=facet, title=f"Actual Plotly Line: {y} vs {x}")
            output_path_html = f"{temp_filename}.html"
            fig.write_html(output_path_html, include_plotlyjs='cdn')
            output_path_png = f"{temp_filename}.png"
            fig.write_image(output_path_png)
            summary_text = f"Debug: Actual Plotly line plot saved to '{output_path_html}' and '{output_path_png}'. Displaying DEBUG chart in UI."
            print(f"[DEBUG data_tools.py] Actual Plotly line plot saved for debugging.")
        except Exception as e_plotly:
            summary_text = f"Debug: Error saving actual Plotly figure: {e_plotly}. Displaying DEBUG chart."
            print(f"[DEBUG data_tools.py] Error saving actual Plotly figure: {e_plotly}")
            
    forced_chart_data = None
    if plot_type == "line": 
        forced_chart_data_payload = ChartJsPayload(
            type="line",
            data=ChartJsData(
                labels=["Debug1", "Debug2", "Debug3"],
                datasets=[ChartJsDataset(label=f"Debug Plot for {dataset_name} ({str(x)} vs {str(y)})", data=[10, 20, 15], borderColor='rgb(255, 99, 132)', fill=False)]
            ),
            options=ChartJsOptions(
                responsive=True,
                scales=ChartJsScales(
                    x=ChartJsScale(type='category', title=ChartJsScaleTitle(display=True, text=str(x) if x else 'X-Axis (Debug)')),
                    y=ChartJsScale(type='linear', beginAtZero=True, title=ChartJsScaleTitle(display=True, text=str(y) if y else 'Y-Axis (Debug)'))
                ),
                plugins=ChartJsPlugins(legend=ChartJsLegend(display=True, position="top"))
            )
        )
        forced_chart_data = forced_chart_data_payload.model_dump() 
        print(f"[DEBUG data_tools.py] Returning FORCED debug chartData for line plot.")
        summary_text = f"Displaying DEBUG line chart for {x} vs {y}. Actual Plotly files also saved if possible."

    return {
        "textSummary": summary_text,
        "chartData": forced_chart_data,
        "filePath_html": output_path_html, 
        "filePath_png": output_path_png
    }

@tool
def run_analysis(dataset_name: str, analysis_type: str) -> str:
    """
    Run a predefined analysis on a dataset.
    Supported analysis_types: summary, correlation, outliers, missing.
    Args:
        dataset_name: Name of the dataset to analyze.
        analysis_type: Type of analysis to run.
    Returns:
        A string with the analysis results or an error message.
    """
    if dataset_name not in DATASETS: return f"Error: Dataset '{dataset_name}' not found."
    df = DATASETS[dataset_name]
    try:
        if analysis_type == "summary":
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty: return "No numeric columns for summary."
            return f"Summary Statistics:\n{numeric_df.describe().to_string()}"
        elif analysis_type == "correlation":
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2: return "Need at least 2 numeric columns for correlation."
            corr_matrix = numeric_df.corr()
            return f"Correlation Matrix:\n{corr_matrix.to_string()}"
        elif analysis_type == "outliers": 
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty: return "No numeric columns for outlier analysis."
            result = "Outlier Analysis (IQR method):\n"
            for column in numeric_df.columns:
                Q1 = numeric_df[column].quantile(0.25); Q3 = numeric_df[column].quantile(0.75); IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR; upper_bound = Q3 + 1.5 * IQR
                outliers = numeric_df[(numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)][column]
                result += f"\nColumn: {column}\n  Outliers: {len(outliers)} ({len(outliers)/len(numeric_df)*100:.1f}%)"
            for column in numeric_df.columns:
                Q1 = numeric_df[column].quantile(0.25); Q3 = numeric_df[column].quantile(0.75); IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR; upper_bound = Q3 + 1.5 * IQR
                outliers = numeric_df[(numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)][column]
                result += f"\nColumn: {column}\n  Outliers: {len(outliers)} ({len(outliers)/len(numeric_df)*100:.1f}%)"
                if not outliers.empty: result += f" Values: {outliers.head().tolist()}{'...' if len(outliers) > 5 else ''}"
            return result
        elif analysis_type == "missing": 
            missing_summary = df.isnull().sum(); missing_summary = missing_summary[missing_summary > 0]
            if missing_summary.empty: return "No missing values found."
            return f"Missing values per column:\n{missing_summary.to_string()}"
        else: return f"Error: Unsupported analysis type '{analysis_type}'. Supported: summary, correlation, outliers, missing."
    except Exception as e: return f"Error in '{analysis_type}' analysis: {str(e)}"

@tool
def generate_sample_data(rows: int = 100, data_type: str = "random") -> str:
    """
    Generate a sample dataset for testing and demonstration purposes.
    Supported data_types: random, sales.
    Args:
        rows: Number of rows to generate.
        data_type: Type of data to generate.
    Returns:
        A message indicating success and the name of the generated dataset.
    """
    try:
        if rows <= 0: return "Error: Number of rows must be positive."
        df_gen = None 
        dataset_name = f"{data_type}_data"
        if data_type == "random":
            df_gen = pd.DataFrame({'id': range(1, rows + 1), 'num_norm': np.random.normal(0,1,rows), 'num_unif': np.random.uniform(0,100,rows), 'int': np.random.randint(1,100,rows), 'cat': np.random.choice(['A','B','C','D'],rows), 'bool': np.random.choice([True,False],rows), 'date': pd.date_range(start='2023-01-01',periods=rows)})
            for col in df_gen.columns[1:]: df_gen.loc[np.random.random(rows) < 0.05, col] = np.nan
        elif data_type == "sales":
            prod = ['A','B','C','D','E']; reg = ['N','S','E','W','C']; chan = ['Online','Retail','Direct']
            df_gen = pd.DataFrame({'date': pd.date_range(start='2023-01-01',periods=rows), 'product': np.random.choice(prod,rows), 'region': np.random.choice(reg,rows), 'channel': np.random.choice(chan,rows), 'units': np.random.randint(1,100,rows), 'price': np.random.uniform(10,1000,rows).round(2), 'cost': np.random.uniform(5,500,rows).round(2)})
            df_gen['revenue'] = df_gen['units'] * df_gen['price']; df_gen['profit'] = df_gen['revenue'] - (df_gen['units'] * df_gen['cost'])
            df_gen.loc[np.random.random(rows) < 0.03, 'units'] = np.nan
        else: return f"Error: Unsupported data type '{data_type}'. Supported: random, sales."
        
        DATASETS[dataset_name] = df_gen
        return f"Successfully generated {rows} rows of {data_type} data as dataset '{dataset_name}' with shape {df_gen.shape}"
    except Exception as e: return f"Error generating sample data: {str(e)}"
