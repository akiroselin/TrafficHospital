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
from langchain.pydantic_v1 import BaseModel, Field
from io import StringIO


# Global variable to store loaded datasets
DATASETS = {}

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
    
    Args:
        file_path: Path to the CSV file to load
        
    Returns:
        A message indicating success or failure
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        
        df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path)
        dataset_name = os.path.splitext(file_name)[0]
        DATASETS[dataset_name] = df
        
        return f"Successfully loaded '{file_path}' as dataset '{dataset_name}' with shape {df.shape}"
    except Exception as e:
        return f"Error loading CSV file: {str(e)}"

@tool
def load_excel(file_path: str, sheet_name: Optional[str] = None) -> str:
    """
    Load an Excel file into a pandas DataFrame and store it in memory.
    
    Args:
        file_path: Path to the Excel file to load
        sheet_name: Optional name of the sheet to load (if None, loads the first sheet)
        
    Returns:
        A message indicating success or failure
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        file_name = os.path.basename(file_path)
        dataset_name = os.path.splitext(file_name)[0]
        DATASETS[dataset_name] = df
        
        return f"Successfully loaded '{file_path}' as dataset '{dataset_name}' with shape {df.shape}"
    except Exception as e:
        return f"Error loading Excel file: {str(e)}"

@tool
def list_datasets() -> str:
    """
    List all datasets currently loaded in memory.
    
    Returns:
        A string listing all available datasets and their shapes
    """
    if not DATASETS:
        return "No datasets currently loaded."
    
    result = "Available datasets:\n"
    for name, df in DATASETS.items():
        result += f"- {name}: shape {df.shape}\n"
    return result

@tool
def get_dataset_info(dataset_name: str) -> str:
    """
    Get detailed information about a dataset.
    
    Args:
        dataset_name: Name of the dataset to get information about
        
    Returns:
        A string with detailed information about the dataset
    """
    if dataset_name not in DATASETS:
        return f"Error: Dataset '{dataset_name}' not found. Use list_datasets() to see available datasets."
    
    df = DATASETS[dataset_name]
    
    # Create a DataFrame info object
    info = DataFrameInfo(
        shape=df.shape,
        columns=df.columns.tolist(),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        head=df.head().to_string(),
        description=df.describe().to_string(),
        missing_values={col: int(df[col].isna().sum()) for col in df.columns}
    )
    
    # Format the output
    result = f"Dataset: {dataset_name}\n"
    result += f"Shape: {info.shape[0]} rows × {info.shape[1]} columns\n\n"
    
    result += "Columns:\n"
    for col, dtype in info.dtypes.items():
        missing = info.missing_values[col]
        missing_pct = missing / info.shape[0] * 100
        result += f"- {col} ({dtype}): {missing} missing values ({missing_pct:.1f}%)\n"
    
    result += f"\nPreview:\n{info.head}\n\n"
    result += f"Summary Statistics:\n{info.description}\n"
    
    return result

@tool
def query_data(dataset_name: str, query: str) -> str:
    """
    Run a query on a dataset using pandas query syntax.
    
    Args:
        dataset_name: Name of the dataset to query
        query: Query string using pandas query syntax
        
    Returns:
        A string representation of the query results
    """
    if dataset_name not in DATASETS:
        return f"Error: Dataset '{dataset_name}' not found. Use list_datasets() to see available datasets."
    
    try:
        df = DATASETS[dataset_name]
        result = df.query(query)
        return f"Query returned {len(result)} rows:\n{result.head(10).to_string()}\n\n" + \
               f"Note: Only showing first 10 rows. Full result has {len(result)} rows."
    except Exception as e:
        return f"Error executing query: {str(e)}"

@tool
def describe_column(dataset_name: str, column_name: str) -> str:
    """
    Get detailed statistics about a specific column in a dataset.
    
    Args:
        dataset_name: Name of the dataset
        column_name: Name of the column to describe
        
    Returns:
        A string with detailed statistics about the column
    """
    if dataset_name not in DATASETS:
        return f"Error: Dataset '{dataset_name}' not found. Use list_datasets() to see available datasets."
    
    df = DATASETS[dataset_name]
    
    if column_name not in df.columns:
        return f"Error: Column '{column_name}' not found in dataset '{dataset_name}'."
    
    try:
        series = df[column_name]
        result = f"Column: {column_name}\n"
        result += f"Data type: {series.dtype}\n"
        result += f"Count: {series.count()} non-null values out of {len(series)} entries\n"
        result += f"Missing values: {series.isna().sum()} ({series.isna().mean() * 100:.1f}%)\n"
        
        if pd.api.types.is_numeric_dtype(series):
            result += f"Min: {series.min()}\n"
            result += f"Max: {series.max()}\n"
            result += f"Mean: {series.mean()}\n"
            result += f"Median: {series.median()}\n"
            result += f"Standard deviation: {series.std()}\n"
            result += f"Quartiles:\n{series.quantile([0.25, 0.5, 0.75]).to_string()}\n"
        elif pd.api.types.is_string_dtype(series):
            result += f"Unique values: {series.nunique()}\n"
            value_counts = series.value_counts().head(10)
            result += f"Top values:\n{value_counts.to_string()}\n"
            if len(value_counts) < series.nunique():
                result += f"Note: Only showing top 10 values out of {series.nunique()} unique values.\n"
        
        return result
    except Exception as e:
        return f"Error describing column: {str(e)}"

@tool
def create_visualization(
    dataset_name: str, 
    plot_type: str, 
    x: Optional[str] = None, 
    y: Optional[str] = None,
    hue: Optional[str] = None,
    filename: Optional[str] = None
) -> str:
    """
    Create a visualization of the data and save it to a file.
    
    Args:
        dataset_name: Name of the dataset to visualize
        plot_type: Type of plot (histogram, scatter, bar, box, line, heatmap, pair)
        x: Column name for x-axis (optional for some plot types)
        y: Column name for y-axis (optional for some plot types)
        hue: Column name for color grouping (optional)
        filename: Name of the file to save the plot to (without extension)
        
    Returns:
        A message indicating success or failure and the path to the saved plot
    """
    if dataset_name not in DATASETS:
        return f"Error: Dataset '{dataset_name}' not found. Use list_datasets() to see available datasets."
    
    df = DATASETS[dataset_name]
    
    # Validate column names
    if x and x not in df.columns:
        return f"Error: Column '{x}' not found in dataset '{dataset_name}'."
    if y and y not in df.columns:
        return f"Error: Column '{y}' not found in dataset '{dataset_name}'."
    if hue and hue not in df.columns:
        return f"Error: Column '{hue}' not found in dataset '{dataset_name}'."
    
    # Set default filename if not provided
    if not filename:
        filename = f"{dataset_name}_{plot_type}"
    
    # Ensure the filename has no extension
    filename = os.path.splitext(filename)[0]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    try:
        if plot_type == "histogram":
            if not x:
                return "Error: Column name for x-axis is required for histogram."
            sns.histplot(data=df, x=x, hue=hue)
            plt.title(f"Histogram of {x}")
            
        elif plot_type == "scatter":
            if not x or not y:
                return "Error: Column names for both x and y axes are required for scatter plot."
            sns.scatterplot(data=df, x=x, y=y, hue=hue)
            plt.title(f"Scatter plot of {y} vs {x}")
            
        elif plot_type == "bar":
            if not x or not y:
                return "Error: Column names for both x and y axes are required for bar plot."
            sns.barplot(data=df, x=x, y=y, hue=hue)
            plt.title(f"Bar plot of {y} by {x}")
            
        elif plot_type == "box":
            if not x or not y:
                return "Error: Column names for both x and y axes are required for box plot."
            sns.boxplot(data=df, x=x, y=y, hue=hue)
            plt.title(f"Box plot of {y} by {x}")
            
        elif plot_type == "line":
            if not x or not y:
                return "Error: Column names for both x and y axes are required for line plot."
            sns.lineplot(data=df, x=x, y=y, hue=hue)
            plt.title(f"Line plot of {y} vs {x}")
            
        elif plot_type == "heatmap":
            if x and y:
                # Create a pivot table for the heatmap
                pivot_data = df.pivot_table(index=x, columns=y, aggfunc='size', fill_value=0)
                sns.heatmap(pivot_data, annot=True, cmap="YlGnBu")
                plt.title(f"Heatmap of {x} vs {y}")
            else:
                # Use correlation matrix if no specific columns provided
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
                plt.title("Correlation Matrix Heatmap")
            
        elif plot_type == "pair":
            columns = [col for col in [x, y] if col]
            if hue:
                columns.append(hue)
            if not columns:
                # If no columns specified, use all numeric columns (up to 5)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                columns = numeric_cols[:5]
            sns.pairplot(df[columns], hue=hue)
            plt.suptitle(f"Pair plot of {', '.join(columns)}", y=1.02)
            
        else:
            return f"Error: Unsupported plot type '{plot_type}'. Supported types: histogram, scatter, bar, box, line, heatmap, pair."
        
        # Save the plot
        output_path = f"{filename}.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return f"Successfully created {plot_type} plot and saved to '{output_path}'."
    except Exception as e:
        plt.close()
        return f"Error creating visualization: {str(e)}"

@tool
def run_analysis(dataset_name: str, analysis_type: str) -> str:
    """
    Run a predefined analysis on a dataset.
    
    Args:
        dataset_name: Name of the dataset to analyze
        analysis_type: Type of analysis to run (summary, correlation, outliers, missing)
        
    Returns:
        A string with the analysis results
    """
    if dataset_name not in DATASETS:
        return f"Error: Dataset '{dataset_name}' not found. Use list_datasets() to see available datasets."
    
    df = DATASETS[dataset_name]
    
    try:
        if analysis_type == "summary":
            # Basic summary statistics
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return "No numeric columns found in the dataset for summary analysis."
            
            result = "Summary Statistics:\n"
            result += numeric_df.describe().to_string()
            return result
            
        elif analysis_type == "correlation":
            # Correlation analysis
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return "No numeric columns found in the dataset for correlation analysis."
            
            corr_matrix = numeric_df.corr()
            result = "Correlation Matrix:\n"
            result += corr_matrix.to_string()
            
            # Highlight strong correlations
            strong_corr = []
            for i, row in enumerate(corr_matrix.values):
                for j, val in enumerate(row):
                    if i < j and abs(val) > 0.7:  # Only upper triangle and strong correlations
                        strong_corr.append((corr_matrix.index[i], corr_matrix.columns[j], val))
            
            if strong_corr:
                result += "\n\nStrong Correlations (|r| > 0.7):\n"
                for var1, var2, corr in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                    result += f"- {var1} and {var2}: {corr:.3f}\n"
            
            return result
            
        elif analysis_type == "outliers":
            # Outlier detection
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return "No numeric columns found in the dataset for outlier analysis."
            
            result = "Outlier Analysis:\n"
            
            for column in numeric_df.columns:
                q1 = numeric_df[column].quantile(0.25)
                q3 = numeric_df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = numeric_df[(numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)][column]
                
                result += f"\nColumn: {column}\n"
                result += f"- IQR: {iqr}\n"
                result += f"- Outlier boundaries: [{lower_bound}, {upper_bound}]\n"
                result += f"- Number of outliers: {len(outliers)} ({len(outliers) / len(numeric_df) * 100:.1f}% of data)\n"
                
                if not outliers.empty:
                    result += f"- Outlier values: {outliers.values[:5]}"
                    if len(outliers) > 5:
                        result += f" ... and {len(outliers) - 5} more"
                    result += "\n"
            
            return result
            
        elif analysis_type == "missing":
            # Missing value analysis
            result = "Missing Value Analysis:\n"
            
            missing = df.isna().sum()
            missing_pct = missing / len(df) * 100
            
            missing_df = pd.DataFrame({
                'Missing Values': missing,
                'Percentage': missing_pct
            })
            
            # Sort by missing percentage
            missing_df = missing_df.sort_values('Percentage', ascending=False)
            
            # Only include columns with missing values
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            
            if missing_df.empty:
                result += "No missing values found in the dataset."
            else:
                result += missing_df.to_string()
                
                # Suggest actions for handling missing values
                result += "\n\nSuggestions for handling missing values:\n"
                for col, row in missing_df.iterrows():
                    pct = row['Percentage']
                    if pct > 50:
                        result += f"- {col}: Consider dropping this column as it has more than 50% missing values.\n"
                    elif pct > 20:
                        result += f"- {col}: Consider imputation techniques or using models that handle missing values.\n"
                    else:
                        result += f"- {col}: Could be imputed with mean/median/mode or dropped rows with missing values.\n"
            
            return result
            
        else:
            return f"Error: Unsupported analysis type '{analysis_type}'. Supported types: summary, correlation, outliers, missing."
            
    except Exception as e:
        return f"Error running analysis: {str(e)}"

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
) -> str:
    """
    Create an interactive visualization of the data with hover information and save it as an HTML file.
    
    Args:
        dataset_name: Name of the dataset to visualize
        plot_type: Type of plot (bar, line, scatter, box, histogram, heatmap, pie, sunburst)
        x: Column name for x-axis (optional for some plot types)
        y: Column name for y-axis (optional for some plot types)
        color: Column name for color encoding (optional)
        size: Column name for size encoding in scatter plots (optional)
        facet: Column name for creating faceted/grouped plots (optional)
        filename: Name of the file to save the plot to (without extension)
        
    Returns:
        A message indicating success or failure and the path to the saved plot
    """
    if dataset_name not in DATASETS:
        return f"Error: Dataset '{dataset_name}' not found. Use list_datasets() to see available datasets."
    
    df = DATASETS[dataset_name]
    
    # Validate column names
    if x and x not in df.columns:
        return f"Error: Column '{x}' not found in dataset '{dataset_name}'."
    if y and y not in df.columns:
        return f"Error: Column '{y}' not found in dataset '{dataset_name}'."
    if color and color not in df.columns:
        return f"Error: Column '{color}' not found in dataset '{dataset_name}'."
    if size and size not in df.columns:
        return f"Error: Column '{size}' not found in dataset '{dataset_name}'."
    if facet and facet not in df.columns:
        return f"Error: Column '{facet}' not found in dataset '{dataset_name}'."
    
    # Set default filename if not provided
    if not filename:
        filename = f"{dataset_name}_{plot_type}_interactive"
    
    # Ensure the filename has no extension
    filename = os.path.splitext(filename)[0]
    
    try:
        fig = None
        
        if plot_type == "bar":
            if not x:
                return "Error: Column name for x-axis is required for bar plot."
            
            if y:
                # Bar chart with values
                fig = px.bar(
                    df, x=x, y=y, color=color,
                    facet_col=facet, 
                    title=f"Bar Chart of {y} by {x}",
                    labels={x: x.replace('_', ' ').title(), y: y.replace('_', ' ').title()},
                    hover_data=df.columns
                )
            else:
                # Count plot (value counts)
                counts = df[x].value_counts().reset_index()
                counts.columns = [x, 'count']
                fig = px.bar(
                    counts, x=x, y='count',
                    title=f"Count of {x}",
                    labels={x: x.replace('_', ' ').title(), 'count': 'Count'},
                    hover_data={x: True, 'count': True}
                )
        
        elif plot_type == "line":
            if not x or not y:
                return "Error: Column names for both x and y axes are required for line plot."
            
            fig = px.line(
                df, x=x, y=y, color=color,
                facet_col=facet,
                title=f"Line Chart of {y} vs {x}",
                labels={x: x.replace('_', ' ').title(), y: y.replace('_', ' ').title()},
                hover_data=df.columns
            )
        
        elif plot_type == "scatter":
            if not x or not y:
                return "Error: Column names for both x and y axes are required for scatter plot."
            
            fig = px.scatter(
                df, x=x, y=y, color=color, size=size,
                facet_col=facet,
                title=f"Scatter Plot of {y} vs {x}",
                labels={
                    x: x.replace('_', ' ').title(), 
                    y: y.replace('_', ' ').title(),
                    color: color.replace('_', ' ').title() if color else None,
                    size: size.replace('_', ' ').title() if size else None
                },
                hover_data=df.columns
            )
            
            # Add trendline if both x and y are numeric
            if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                fig.update_layout(
                    shapes=[{
                        'type': 'line',
                        'x0': df[x].min(),
                        'y0': df[y].min(),
                        'x1': df[x].max(),
                        'y1': df[y].max(),
                        'line': {
                            'color': 'red',
                            'width': 1,
                            'dash': 'dash'
                        }
                    }]
                )
        
        elif plot_type == "box":
            if not y:
                return "Error: Column name for y-axis is required for box plot."
            
            fig = px.box(
                df, x=x, y=y, color=color,
                facet_col=facet,
                title=f"Box Plot of {y}" + (f" by {x}" if x else ""),
                labels={
                    x: x.replace('_', ' ').title() if x else None, 
                    y: y.replace('_', ' ').title()
                },
                hover_data=df.columns
            )
        
        elif plot_type == "histogram":
            if not x:
                return "Error: Column name for x-axis is required for histogram."
            
            fig = px.histogram(
                df, x=x, color=color,
                facet_col=facet,
                title=f"Histogram of {x}",
                labels={x: x.replace('_', ' ').title()},
                hover_data=df.columns,
                marginal="box"  # Add box plot on the margin
            )
        
        elif plot_type == "heatmap":
            if not x or not y:
                # Use correlation matrix if no specific columns provided
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                
                # Create heatmap using plotly graph objects for more control
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1,
                    text=corr_matrix.round(2).values,
                    hovertemplate='%{y} & %{x}<br>Correlation: %{z:.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Correlation Matrix Heatmap",
                    xaxis_title="Features",
                    yaxis_title="Features"
                )
            else:
                # Create a pivot table for the heatmap
                if pd.api.types.is_numeric_dtype(df[y]):
                    # If y is numeric, use it as values
                    pivot_data = df.pivot_table(index=x, columns=color if color else None, values=y, aggfunc='mean')
                    title = f"Heatmap of Average {y} by {x}" + (f" and {color}" if color else "")
                else:
                    # Otherwise use counts
                    pivot_data = pd.crosstab(df[x], df[y])
                    title = f"Heatmap of {x} vs {y} (Counts)"
                
                # Create heatmap
                fig = px.imshow(
                    pivot_data,
                    labels=dict(x=y, y=x, color=y),
                    title=title,
                    text_auto=True,
                    aspect="auto"
                )
        
        elif plot_type == "pie":
            if not x:
                return "Error: Column name for values is required for pie chart."
            
            # Group by the column and count or sum
            if y and pd.api.types.is_numeric_dtype(df[y]):
                # If y is provided and numeric, use it for values
                grouped = df.groupby(x)[y].sum().reset_index()
                values = y
                title = f"Pie Chart of Total {y} by {x}"
            else:
                # Otherwise use counts
                grouped = df[x].value_counts().reset_index()
                grouped.columns = [x, 'count']
                values = 'count'
                title = f"Pie Chart of {x} Distribution"
            
            fig = px.pie(
                grouped, names=x, values=values,
                title=title,
                hover_data=[x, values],
                labels={x: x.replace('_', ' ').title(), values: values.replace('_', ' ').title()}
            )
        
        elif plot_type == "sunburst":
            if not x:
                return "Error: At least one column name is required for sunburst chart."
            
            # Prepare path for sunburst
            path = [x]
            if color:
                path.append(color)
            if facet:
                path.append(facet)
            
            # Determine values
            if y and pd.api.types.is_numeric_dtype(df[y]):
                values = y
                title = f"Sunburst Chart of {y} by {', '.join(path)}"
            else:
                values = None  # Will use counts
                title = f"Sunburst Chart of {', '.join(path)}"
            
            fig = px.sunburst(
                df, path=path, values=values,
                title=title,
                hover_data=df.columns
            )
        
        else:
            return f"Error: Unsupported plot type '{plot_type}'. Supported types: bar, line, scatter, box, histogram, heatmap, pie, sunburst."
        
        # Add common layout improvements
        fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Save the plot as HTML for interactivity
        output_path = f"{filename}.html"
        fig.write_html(output_path, include_plotlyjs='cdn')
        
        # Also save as PNG for easy viewing
        png_path = f"{filename}.png"
        fig.write_image(png_path)
        
        return f"Successfully created interactive {plot_type} plot and saved to '{output_path}' (HTML) and '{png_path}' (PNG)."
    except Exception as e:
        return f"Error creating interactive visualization: {str(e)}"

@tool
def generate_sample_data(rows: int = 100, data_type: str = "random") -> str:
    """
    Generate a sample dataset for testing and demonstration purposes.
    
    Args:
        rows: Number of rows to generate (default: 100)
        data_type: Type of data to generate (random, sales, weather, stocks)
        
    Returns:
        A message indicating success and the name of the generated dataset
    """
    try:
        if rows <= 0:
            return "Error: Number of rows must be positive."
        
        if data_type == "random":
            # Generate random data with various data types
            df = pd.DataFrame({
                'id': range(1, rows + 1),
                'numeric_normal': np.random.normal(0, 1, rows),
                'numeric_uniform': np.random.uniform(0, 100, rows),
                'integer': np.random.randint(1, 100, rows),
                'category': np.random.choice(['A', 'B', 'C', 'D'], rows),
                'boolean': np.random.choice([True, False], rows),
                'date': pd.date_range(start='2023-01-01', periods=rows)
            })
            
            # Add some missing values
            for col in df.columns[1:]:  # Skip the ID column
                mask = np.random.random(rows) < 0.05  # 5% missing values
                df.loc[mask, col] = np.nan
                
            dataset_name = "random_data"
            
        elif data_type == "sales":
            # Generate sales data
            products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
            regions = ['North', 'South', 'East', 'West', 'Central']
            channels = ['Online', 'Retail', 'Direct', 'Distributor']
            
            df = pd.DataFrame({
                'date': pd.date_range(start='2023-01-01', periods=rows),
                'product': np.random.choice(products, rows),
                'region': np.random.choice(regions, rows),
                'channel': np.random.choice(channels, rows),
                'units_sold': np.random.randint(1, 100, rows),
                'unit_price': np.random.uniform(10, 1000, rows).round(2),
                'cost': np.random.uniform(5, 500, rows).round(2)
            })
            
            # Calculate revenue and profit
            df['revenue'] = df['units_sold'] * df['unit_price']
            df['profit'] = df['revenue'] - (df['units_sold'] * df['cost'])
            
            # Add some missing values
            mask = np.random.random(rows) < 0.03  # 3% missing values
            df.loc[mask, 'units_sold'] = np.nan
            
            dataset_name = "sales_data"
            
        elif data_type == "weather":
            # Generate weather data
            cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego']
            
            df = pd.DataFrame({
                'date': pd.date_range(start='2023-01-01', periods=rows),
                'city': np.random.choice(cities, rows),
                'temperature': np.random.normal(15, 10, rows).round(1),  # in Celsius
                'humidity': np.random.uniform(30, 90, rows).round(1),  # in percentage
                'wind_speed': np.random.exponential(5, rows).round(1),  # in km/h
                'pressure': np.random.normal(1013, 10, rows).round(1),  # in hPa
                'precipitation': np.random.exponential(2, rows).round(1)  # in mm
            })
            
            # Add some seasonality to temperature
            day_of_year = df['date'].dt.dayofyear
            seasonal_effect = 15 * np.sin(2 * np.pi * day_of_year / 365)
            df['temperature'] = df['temperature'] + seasonal_effect
            
            # Add some missing values
            for col in ['humidity', 'wind_speed', 'pressure', 'precipitation']:
                mask = np.random.random(rows) < 0.04  # 4% missing values
                df.loc[mask, col] = np.nan
                
            dataset_name = "weather_data"
            
        elif data_type == "stocks":
            # Generate stock market data
            stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']
            
            # Base dataframe with dates and stock symbols
            dates = pd.date_range(start='2023-01-01', periods=rows // len(stocks) + 1)
            stocks_repeated = np.repeat(stocks, len(dates))[:rows]
            dates_repeated = np.tile(dates, len(stocks))[:rows]
            
            df = pd.DataFrame({
                'date': dates_repeated,
                'symbol': stocks_repeated
            })
            
            # Generate price data with random walk
            symbols = df['symbol'].unique()
            start_prices = {
                'AAPL': 150, 'MSFT': 250, 'GOOGL': 100, 'AMZN': 120, 
                'META': 200, 'TSLA': 180, 'NVDA': 300, 'JPM': 140
            }
            
            prices = []
            volumes = []
            
            for symbol in symbols:
                symbol_rows = df[df['symbol'] == symbol].shape[0]
                start_price = start_prices.get(symbol, 100)
                
                # Random walk for price
                daily_returns = np.random.normal(0.0005, 0.015, symbol_rows)
                price_series = start_price * (1 + daily_returns).cumprod()
                
                # Volume with some randomness
                avg_volume = np.random.randint(100000, 10000000)
                volume_series = np.random.normal(avg_volume, avg_volume * 0.3, symbol_rows).astype(int)
                volume_series = np.maximum(volume_series, 0)  # Ensure non-negative
                
                prices.extend(price_series)
                volumes.extend(volume_series)
            
            df['open'] = prices
            df['high'] = df['open'] * (1 + np.random.uniform(0, 0.03, rows))
            df['low'] = df['open'] * (1 - np.random.uniform(0, 0.03, rows))
            df['close'] = df['low'] + np.random.uniform(0, 1, rows) * (df['high'] - df['low'])
            df['volume'] = volumes
            
            # Round price columns to 2 decimal places
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].round(2)
                
            dataset_name = "stock_data"
            
        else:
            return f"Error: Unsupported data type '{data_type}'. Supported types: random, sales, weather, stocks."
        
        # Store the dataset
        DATASETS[dataset_name] = df
        
        return f"Successfully generated {rows} rows of {data_type} data as dataset '{dataset_name}' with shape {df.shape}"
    except Exception as e:
        return f"Error generating sample data: {str(e)}"
