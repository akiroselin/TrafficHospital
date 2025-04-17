"""
Example script demonstrating how to use the Data Analysis Agent programmatically.

This script shows how to:
1. Initialize the agent
2. Generate sample data
3. Run specific analyses
4. Create visualizations
5. Ask complex questions

You can use this as a template for your own scripts.
"""

import os
from dotenv import load_dotenv
from main import create_data_analysis_agent
from data_tools import generate_sample_data as generate_sample_data_tool

# Load environment variables
load_dotenv()

def run_example():
    """Run an example workflow with the data analysis agent."""
    
    print("Initializing Data Analysis Agent...")
    agent = create_data_analysis_agent(verbose=True)
    
    print("\n" + "="*50)
    print("EXAMPLE 1: Generate and analyze sample data")
    print("="*50)
    
    # Step 1: Generate sample stock market data
    response = agent.invoke({
        "input": "Generate a sample stock market dataset with 200 rows"
    })
    print(f"\nAgent Response:\n{response['output']}\n")
    
    # Step 2: Get information about the dataset
    response = agent.invoke({
        "input": "Tell me about the stock_data dataset"
    })
    print(f"\nAgent Response:\n{response['output']}\n")
    
    # Step 3: Run correlation analysis
    response = agent.invoke({
        "input": "Run a correlation analysis on the stock_data"
    })
    print(f"\nAgent Response:\n{response['output']}\n")
    
    # Step 4: Create a visualization
    response = agent.invoke({
        "input": "Create a line plot of close vs date for AAPL in the stock_data"
    })
    print(f"\nAgent Response:\n{response['output']}\n")
    
    print("\n" + "="*50)
    print("EXAMPLE 2: Generate and analyze sales data")
    print("="*50)
    
    # Step 1: Generate sample sales data
    response = agent.invoke({
        "input": "Generate a sample sales dataset with 150 rows"
    })
    print(f"\nAgent Response:\n{response['output']}\n")
    
    # Step 2: Ask a complex question
    response = agent.invoke({
        "input": "Which product has the highest average profit margin, and how does it vary by region?"
    })
    print(f"\nAgent Response:\n{response['output']}\n")
    
    # Step 3: Create a visualization based on the analysis
    response = agent.invoke({
        "input": "Create a bar chart showing profit by product and region"
    })
    print(f"\nAgent Response:\n{response['output']}\n")
    
    # Step 4: Find outliers
    response = agent.invoke({
        "input": "Find outliers in the profit column of the sales_data"
    })
    print(f"\nAgent Response:\n{response['output']}\n")
    
    print("\n" + "="*50)
    print("EXAMPLE 3: Custom multi-step analysis")
    print("="*50)
    
    # Run a multi-step analysis with a single prompt
    response = agent.invoke({
        "input": """
        For the sales_data:
        1. First, show me the distribution of sales by channel
        2. Then analyze if there's a correlation between unit_price and units_sold
        3. Finally, tell me which region has the highest average revenue and why
        """
    })
    print(f"\nAgent Response:\n{response['output']}\n")

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
