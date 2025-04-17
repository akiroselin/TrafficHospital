"""
Example script demonstrating how to extend the Data Analysis Agent with custom tools.

This script shows how to:
1. Create custom data analysis tools
2. Register them with the agent
3. Use the extended agent with the new tools

You can use this as a template for adding your own custom tools.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory

# Import our existing tools and agent creator
from data_tools import DATASETS
from main import create_data_analysis_agent

# Load environment variables
load_dotenv()

# Define custom tools
@tool
def calculate_roi(dataset_name: str, investment_column: str, return_column: str) -> str:
    """
    Calculate Return on Investment (ROI) for each row in a dataset.
    
    Args:
        dataset_name: Name of the dataset to analyze
        investment_column: Column name containing investment values
        return_column: Column name containing return values
        
    Returns:
        A string with ROI analysis results
    """
    if dataset_name not in DATASETS:
        return f"Error: Dataset '{dataset_name}' not found. Use list_datasets() to see available datasets."
    
    df = DATASETS[dataset_name]
    
    # Validate column names
    if investment_column not in df.columns:
        return f"Error: Column '{investment_column}' not found in dataset '{dataset_name}'."
    if return_column not in df.columns:
        return f"Error: Column '{return_column}' not found in dataset '{dataset_name}'."
    
    try:
        # Calculate ROI: (Return - Investment) / Investment
        df['roi'] = (df[return_column] - df[investment_column]) / df[investment_column]
        
        # Format as percentage
        df['roi_percent'] = df['roi'] * 100
        
        # Calculate summary statistics
        avg_roi = df['roi'].mean() * 100
        median_roi = df['roi'].median() * 100
        min_roi = df['roi'].min() * 100
        max_roi = df['roi'].max() * 100
        
        # Get top 5 and bottom 5 ROIs
        top_5 = df.sort_values('roi', ascending=False).head(5)
        bottom_5 = df.sort_values('roi', ascending=True).head(5)
        
        # Format the result
        result = "ROI Analysis Results:\n\n"
        result += f"Average ROI: {avg_roi:.2f}%\n"
        result += f"Median ROI: {median_roi:.2f}%\n"
        result += f"Min ROI: {min_roi:.2f}%\n"
        result += f"Max ROI: {max_roi:.2f}%\n\n"
        
        result += "Top 5 ROIs:\n"
        for _, row in top_5.iterrows():
            result += f"- {row.get('product', 'Item')} ({row.get('category', 'Category')}): {row['roi_percent']:.2f}%\n"
        
        result += "\nBottom 5 ROIs:\n"
        for _, row in bottom_5.iterrows():
            result += f"- {row.get('product', 'Item')} ({row.get('category', 'Category')}): {row['roi_percent']:.2f}%\n"
        
        # Create a histogram of ROI distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['roi_percent'], bins=20, alpha=0.7, color='blue')
        plt.axvline(avg_roi, color='red', linestyle='dashed', linewidth=1, label=f'Mean ROI: {avg_roi:.2f}%')
        plt.axvline(median_roi, color='green', linestyle='dashed', linewidth=1, label=f'Median ROI: {median_roi:.2f}%')
        plt.title('ROI Distribution')
        plt.xlabel('ROI (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        output_path = f"{dataset_name}_roi_distribution.png"
        plt.savefig(output_path)
        plt.close()
        
        result += f"\nA histogram of the ROI distribution has been saved to '{output_path}'."
        
        return result
    except Exception as e:
        return f"Error calculating ROI: {str(e)}"

@tool
def segment_data(
    dataset_name: str, 
    column_name: str, 
    num_segments: int = 5,
    segment_type: str = "equal_width"
) -> str:
    """
    Segment data into groups based on a specific column.
    
    Args:
        dataset_name: Name of the dataset to segment
        column_name: Column name to use for segmentation
        num_segments: Number of segments to create (default: 5)
        segment_type: Type of segmentation - 'equal_width', 'equal_size', or 'custom' (default: equal_width)
        
    Returns:
        A string with segmentation analysis results
    """
    if dataset_name not in DATASETS:
        return f"Error: Dataset '{dataset_name}' not found. Use list_datasets() to see available datasets."
    
    df = DATASETS[dataset_name]
    
    # Validate column name
    if column_name not in df.columns:
        return f"Error: Column '{column_name}' not found in dataset '{dataset_name}'."
    
    try:
        # Create a copy of the dataframe to avoid modifying the original
        analysis_df = df.copy()
        
        # Perform segmentation
        if segment_type == "equal_width":
            # Equal width bins (same range size)
            analysis_df['segment'] = pd.cut(
                analysis_df[column_name], 
                bins=num_segments, 
                labels=[f"Segment {i+1}" for i in range(num_segments)]
            )
        elif segment_type == "equal_size":
            # Equal size bins (same number of items)
            analysis_df['segment'] = pd.qcut(
                analysis_df[column_name], 
                q=num_segments, 
                labels=[f"Segment {i+1}" for i in range(num_segments)],
                duplicates='drop'
            )
        else:
            return f"Error: Unsupported segment_type '{segment_type}'. Use 'equal_width' or 'equal_size'."
        
        # Calculate segment statistics
        segment_stats = analysis_df.groupby('segment').agg({
            column_name: ['count', 'min', 'max', 'mean', 'median', 'std']
        })
        
        # Format the result
        result = f"Data Segmentation Results for '{column_name}':\n\n"
        result += segment_stats.to_string() + "\n\n"
        
        # Calculate additional statistics for each segment
        result += "Additional segment statistics:\n\n"
        
        for segment in analysis_df['segment'].unique():
            segment_df = analysis_df[analysis_df['segment'] == segment]
            result += f"{segment}:\n"
            result += f"- Count: {len(segment_df)} records ({len(segment_df) / len(analysis_df) * 100:.1f}% of total)\n"
            result += f"- Range: {segment_df[column_name].min()} to {segment_df[column_name].max()}\n"
            
            # If we have categorical columns, show distribution
            categorical_cols = segment_df.select_dtypes(include=['object', 'category']).columns
            for cat_col in categorical_cols:
                if cat_col != 'segment' and len(segment_df[cat_col].unique()) < 10:
                    dist = segment_df[cat_col].value_counts(normalize=True) * 100
                    result += f"- {cat_col} distribution:\n"
                    for val, pct in dist.items():
                        result += f"  * {val}: {pct:.1f}%\n"
            
            result += "\n"
        
        # Create a visualization of the segments
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Segment sizes
        plt.subplot(1, 2, 1)
        segment_counts = analysis_df['segment'].value_counts().sort_index()
        segment_counts.plot(kind='bar', color='skyblue')
        plt.title('Segment Sizes')
        plt.xlabel('Segment')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Plot 2: Box plot of the segmented variable
        plt.subplot(1, 2, 2)
        sns_available = True
        try:
            import seaborn as sns
            sns.boxplot(x='segment', y=column_name, data=analysis_df)
        except ImportError:
            sns_available = False
            plt.boxplot([analysis_df[analysis_df['segment'] == seg][column_name] for seg in analysis_df['segment'].unique()])
            plt.xticks(range(1, len(analysis_df['segment'].unique()) + 1), analysis_df['segment'].unique())
        
        plt.title(f'Distribution of {column_name} by Segment')
        plt.xlabel('Segment')
        plt.ylabel(column_name)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = f"{dataset_name}_{column_name}_segmentation.png"
        plt.savefig(output_path)
        plt.close()
        
        result += f"A visualization of the segments has been saved to '{output_path}'."
        
        return result
    except Exception as e:
        return f"Error segmenting data: {str(e)}"

@tool
def predict_values(
    dataset_name: str,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2
) -> str:
    """
    Build a simple predictive model for a target column based on other features.
    
    Args:
        dataset_name: Name of the dataset to use
        target_column: Column name to predict
        feature_columns: List of column names to use as features (if None, uses all numeric columns)
        test_size: Proportion of data to use for testing (default: 0.2)
        
    Returns:
        A string with model performance results
    """
    if dataset_name not in DATASETS:
        return f"Error: Dataset '{dataset_name}' not found. Use list_datasets() to see available datasets."
    
    df = DATASETS[dataset_name]
    
    # Validate column names
    if target_column not in df.columns:
        return f"Error: Target column '{target_column}' not found in dataset '{dataset_name}'."
    
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                return f"Error: Feature column '{col}' not found in dataset '{dataset_name}'."
    
    try:
        # Import required libraries
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
        
        # Create a copy of the dataframe to avoid modifying the original
        analysis_df = df.copy()
        
        # Handle missing values
        analysis_df = analysis_df.dropna(subset=[target_column])
        
        # Determine if this is a regression or classification task
        is_numeric_target = pd.api.types.is_numeric_dtype(analysis_df[target_column])
        unique_values = analysis_df[target_column].nunique()
        is_classification = not is_numeric_target or (is_numeric_target and unique_values < 10)
        
        # Select features
        if feature_columns:
            X = analysis_df[feature_columns]
        else:
            # Use all numeric columns except the target
            numeric_cols = analysis_df.select_dtypes(include=['number']).columns
            X = analysis_df[[col for col in numeric_cols if col != target_column]]
        
        # Handle categorical features
        X = pd.get_dummies(X, drop_first=True)
        
        # Get target
        y = analysis_df[target_column]
        if is_classification and is_numeric_target:
            y = y.astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        result = f"Predictive Modeling Results for '{target_column}':\n\n"
        
        if is_classification:
            # Classification task
            result += "Task Type: Classification\n\n"
            
            # Logistic Regression
            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            lr_model.fit(X_train_scaled, y_train)
            lr_preds = lr_model.predict(X_test_scaled)
            lr_accuracy = accuracy_score(y_test, lr_preds)
            
            # Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)  # RF doesn't require scaling
            rf_preds = rf_model.predict(X_test)
            rf_accuracy = accuracy_score(y_test, rf_preds)
            
            # Model comparison
            result += "Model Performance:\n"
            result += f"- Logistic Regression Accuracy: {lr_accuracy:.4f}\n"
            result += f"- Random Forest Accuracy: {rf_accuracy:.4f}\n\n"
            
            # Use the best model for further analysis
            if rf_accuracy >= lr_accuracy:
                best_model = rf_model
                best_preds = rf_preds
                best_model_name = "Random Forest"
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False)
            else:
                best_model = lr_model
                best_preds = lr_preds
                best_model_name = "Logistic Regression"
                # For logistic regression, use coefficients as feature importance
                if len(lr_model.classes_) == 2:  # Binary classification
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': np.abs(best_model.coef_[0])
                    }).sort_values('Importance', ascending=False)
                else:  # Multi-class
                    # Average the absolute coefficients across all classes
                    importance = np.mean(np.abs(best_model.coef_), axis=0)
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False)
            
            # Detailed report for best model
            result += f"Detailed Classification Report ({best_model_name}):\n"
            result += classification_report(y_test, best_preds) + "\n\n"
            
        else:
            # Regression task
            result += "Task Type: Regression\n\n"
            
            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            lr_preds = lr_model.predict(X_test_scaled)
            lr_mse = mean_squared_error(y_test, lr_preds)
            lr_rmse = np.sqrt(lr_mse)
            lr_r2 = r2_score(y_test, lr_preds)
            
            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)  # RF doesn't require scaling
            rf_preds = rf_model.predict(X_test)
            rf_mse = mean_squared_error(y_test, rf_preds)
            rf_rmse = np.sqrt(rf_mse)
            rf_r2 = r2_score(y_test, rf_preds)
            
            # Model comparison
            result += "Model Performance:\n"
            result += f"- Linear Regression RMSE: {lr_rmse:.4f}, R²: {lr_r2:.4f}\n"
            result += f"- Random Forest RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}\n\n"
            
            # Use the best model for further analysis
            if rf_r2 >= lr_r2:
                best_model = rf_model
                best_preds = rf_preds
                best_model_name = "Random Forest"
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False)
            else:
                best_model = lr_model
                best_preds = lr_preds
                best_model_name = "Linear Regression"
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': np.abs(best_model.coef_)
                }).sort_values('Importance', ascending=False)
        
        # Feature importance
        result += "Top 10 Most Important Features:\n"
        for i, (feature, importance) in enumerate(zip(feature_importance['Feature'].head(10), feature_importance['Importance'].head(10))):
            result += f"{i+1}. {feature}: {importance:.4f}\n"
        
        # Create visualizations
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Feature importance
        plt.subplot(2, 1, 1)
        top_features = feature_importance.head(10)
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.title(f'Top 10 Feature Importance ({best_model_name})')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()  # Display the highest importance at the top
        
        # Plot 2: Actual vs Predicted
        plt.subplot(2, 1, 2)
        plt.scatter(y_test, best_preds, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = f"{dataset_name}_{target_column}_prediction.png"
        plt.savefig(output_path)
        plt.close()
        
        result += f"\nA visualization of the model results has been saved to '{output_path}'."
        
        return result
    except Exception as e:
        return f"Error building predictive model: {str(e)}"

def create_extended_agent(
    model_name: Optional[str] = None,
    temperature: float = 0,
    verbose: bool = True
) -> AgentExecutor:
    """
    Create a data analysis agent with custom tools.
    
    Args:
        model_name: Name of the Ollama model to use (defaults to OLLAMA_MODEL env var)
        temperature: Temperature parameter for the model
        verbose: Whether to print verbose output
        
    Returns:
        An AgentExecutor instance with custom tools
    """
    # Check if Ollama configuration is set
    if not os.getenv("OLLAMA_BASE_URL"):
        raise ValueError(
            "OLLAMA_BASE_URL environment variable not set. "
            "Please set it in a .env file or export it as an environment variable."
        )
    
    # Use model from environment variable if not specified
    if model_name is None:
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
    
    # Initialize the language model
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=os.getenv("OLLAMA_BASE_URL")
    )
    
    # Import standard tools
    from data_tools import (
        load_csv, load_excel, list_datasets, get_dataset_info, 
        query_data, describe_column, create_visualization, 
        create_interactive_visualization, run_analysis, generate_sample_data
    )
    
    # Define the tools available to the agent, including our custom tools
    tools = [
        load_csv,
        load_excel,
        list_datasets,
        get_dataset_info,
        query_data,
        describe_column,
        create_visualization,
        create_interactive_visualization,
        run_analysis,
        generate_sample_data,
        # Custom tools
        calculate_roi,
        segment_data,
        predict_values
    ]
    
    # Create a system message that instructs the agent on its capabilities
    system_message = """
    You are a data analysis assistant that helps users analyze and visualize data.
    You have access to several tools that can help you with this task:
    
    Standard tools:
    - load_csv: Load a CSV file into memory
    - load_excel: Load an Excel file into memory
    - list_datasets: List all datasets currently loaded in memory
    - get_dataset_info: Get detailed information about a dataset
    - query_data: Run a query on a dataset using pandas query syntax
    - describe_column: Get detailed statistics about a specific column
    - create_visualization: Create a static visualization of the data
    - create_interactive_visualization: Create an interactive visualization with hover information
    - run_analysis: Run a predefined analysis on a dataset
    - generate_sample_data: Generate a sample dataset for testing
    
    Advanced tools:
    - calculate_roi: Calculate Return on Investment (ROI) for each row in a dataset
    - segment_data: Segment data into groups based on a specific column
    - predict_values: Build a simple predictive model for a target column
    
    When a user asks you to analyze data, first check if the data is already loaded.
    If not, ask the user to provide a file path or generate sample data.
    
    Always explain your analysis in a clear and concise manner.
    When creating visualizations or running analyses, explain what the results show and any insights that can be derived from them.
    
    Remember that you can only work with datasets that have been loaded into memory.
    Use the list_datasets tool to see what datasets are available.
    """
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create a memory object to store conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        handle_parsing_errors=True
    )
    
    return agent_executor

def run_example():
    """Run an example workflow with the extended data analysis agent."""
    
    print("Initializing Extended Data Analysis Agent...")
    agent = create_extended_agent(verbose=True)
    
    print("\n" + "="*50)
    print("EXAMPLE: Using custom tools with the agent")
    print("="*50)
    
    # Step 1: Load sample data
    print("\nStep 1: Loading sample data...")
    from data_tools import load_csv
    result = load_csv.run({"file_path": "sample_data.csv"})
    print(result)
    
    # Step 2: Calculate ROI using our custom tool
    print("\nStep 2: Calculating ROI...")
    response = agent.invoke({
        "input": "Calculate the ROI for each product in the sample_data dataset using cost as investment and profit as return"
    })
    print(f"\nAgent Response:\n{response['output']}\n")
    
    # Step 3: Segment the data
    print("\nStep 3: Segmenting the data...")
    response = agent.invoke({
        "input": "Segment the sample_data dataset by price into 4 equal-width segments"
    })
    print(f"\nAgent Response:\n{response['output']}\n")
    
    # Step 4: Build a predictive model
    print("\nStep 4: Building a predictive model...")
    response = agent.invoke({
        "input": "Build a model to predict profit based on price, cost, and units in the sample_data dataset"
    })
    print(f"\nAgent Response:\n{response['output']}\n")
    
    print("\nExample completed! You can now use the extended agent with your own data.")

if __name__ == "__main__":
    # Check if Ollama configuration is set
    if not os.getenv("OLLAMA_BASE_URL"):
        print("Error: OLLAMA_BASE_URL environment variable not set.")
        print("Please create a .env file with your Ollama configuration or set it as an environment variable.")
        print("Example .env file content:")
        print("OLLAMA_BASE_URL=http://localhost:11434")
        print("OLLAMA_MODEL=llama3")
    else:
        run_example()
