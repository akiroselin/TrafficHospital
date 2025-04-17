"""
Data Analysis Agent using LangChain

This script creates a data analysis agent that can load, analyze, and visualize data
using various tools implemented in the data_tools module.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool

# Import our custom data tools
from data_tools import (
    load_csv, load_excel, list_datasets, get_dataset_info, 
    query_data, describe_column, create_visualization, 
    create_interactive_visualization, run_analysis, generate_sample_data
)

# Load environment variables from .env file
load_dotenv()

def create_data_analysis_agent(
    model_name: Optional[str] = None,
    temperature: float = 0,
    verbose: bool = True
) -> AgentExecutor:
    """
    Create a data analysis agent with the specified model and tools.
    
    Args:
        model_name: Name of the Ollama model to use (defaults to OLLAMA_MODEL env var)
        temperature: Temperature parameter for the model
        verbose: Whether to print verbose output
        
    Returns:
        An AgentExecutor instance
    """
    # Check if Ollama configuration is set
    # Validate Ollama configuration and model compatibility
    required_env_vars = ["OLLAMA_BASE_URL"]
    missing_vars = [v for v in required_env_vars if not os.getenv(v)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            "Please set these in a .env file or as environment variables."
        )
    
    # Use model from environment variable if not specified
    if model_name is None:
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
    
    # Initialize the language model
    # Handle unsupported models with graceful fallback
    try:
        llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL")
        )
        # Test model compatibility
        llm.invoke([{"role": "system", "content": "Test connection"}])
    except Exception as e:
        supported_models = ["llama2", "llama3", "mistral", "falcon"]
        recommended_model = supported_models[0]
        
        return f"Error: Model '{model_name}' is incompatible or unavailable. Supported models include: {', '.join(supported_models)}. Using '{recommended_model}' as fallback."
    
    # Define the tools available to the agent
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
        generate_sample_data
    ]
    
    # Create a system message that instructs the agent on its capabilities
    system_message = """
    You are a data analysis assistant that helps users analyze and visualize data.
    You have access to several tools that can help you with this task:
    
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
    
    When a user asks you to analyze data, first check if the data is already loaded.
    If not, ask the user to provide a file path or generate sample data.
    
    Always explain your analysis in a clear and concise manner.
    When creating visualizations, explain what the visualization shows and any insights that can be derived from it.
    
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

def main():
    """
    Main function to run the data analysis agent interactively.
    """
    print("Initializing Data Analysis Agent...")
    
    try:
        agent = create_data_analysis_agent()
        print("\nData Analysis Agent is ready! Type 'exit' to quit.\n")
        
        # Generate sample data by default to make it easier to get started
        print("Generating sample data for demonstration purposes...")
        result = generate_sample_data.run({"rows": 100, "data_type": "sales"})
        print(result)
        print("\nYou can now ask questions about the data or load your own datasets.")
        print("Example: 'Show me a summary of the sales data' or 'Create a bar chart of revenue by region'\n")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            
            response = agent.invoke({"input": user_input})
            print(f"\nAgent: {response['output']}\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        if "OLLAMA_BASE_URL" in str(e):
            print("\nPlease create a .env file with your OLLAMA_BASE_URL or set it as an environment variable.")
            print("Example .env file content:")
            print("OLLAMA_BASE_URL=http://localhost:11434")
            print("OLLAMA_MODEL=llama3")

if __name__ == "__main__":
    main()
