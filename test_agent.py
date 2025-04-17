"""
Test script to verify that the Data Analysis Agent is working correctly.

This script performs basic tests to ensure that:
1. The environment is set up correctly
2. The agent can be initialized
3. The data tools are functioning properly
4. The agent can respond to simple queries

Run this script to verify your setup before using the agent for real tasks.
"""

import os
import sys
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

def run_tests():
    """Run a series of tests to verify the agent setup."""
    
    print("Running Data Analysis Agent tests...\n")
    
    # Test 1: Check environment
    print("Test 1: Checking environment...")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"  - Python version: {python_version}")
    
    # Check required packages
    required_packages = [
        "langchain", "langchain_ollama", "pandas", 
        "numpy", "matplotlib", "seaborn", "scikit-learn"
    ]
    
    all_packages_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  - {package}: Installed")
        except ImportError:
            print(f"  - {package}: NOT INSTALLED")
            all_packages_installed = False
    
    # Check Ollama configuration
    base_url = os.getenv("OLLAMA_BASE_URL")
    model = os.getenv("OLLAMA_MODEL")
    if base_url:
        print(f"  - OLLAMA_BASE_URL: Found ({base_url})")
        if model:
            print(f"  - OLLAMA_MODEL: Found ({model})")
        else:
            print("  - OLLAMA_MODEL: NOT FOUND (will use default 'llama3')")
    else:
        print("  - OLLAMA_BASE_URL: NOT FOUND")
        print("    Please set your Ollama configuration in a .env file or as environment variables.")
        print("    Example: OLLAMA_BASE_URL=http://localhost:11434")
        print("             OLLAMA_MODEL=llama3")
        return False
    
    if not all_packages_installed:
        print("\nSome required packages are missing. Please install them using:")
        print("pip install -r requirements.txt")
        return False
    
    print("  ✓ Environment check passed\n")
    
    # Test 2: Import agent and tools
    print("Test 2: Importing agent and tools...")
    try:
        from main import create_data_analysis_agent
        from data_tools import (
            load_csv, load_excel, list_datasets, get_dataset_info, 
            query_data, describe_column, create_visualization, 
            run_analysis, generate_sample_data, DATASETS
        )
        print("  ✓ Successfully imported agent and tools\n")
    except ImportError as e:
        print(f"  ✗ Import error: {str(e)}")
        print("    Please make sure main.py and data_tools.py are in the current directory.")
        return False
    
    # Test 3: Test data tools directly
    print("Test 3: Testing data tools directly...")
    
    # Generate sample data
    try:
        result = generate_sample_data(rows=10, data_type="random")
        print(f"  - generate_sample_data: {result}")
        
        # List datasets
        result = list_datasets()
        print(f"  - list_datasets: {result}")
        
        # Get dataset info
        result = get_dataset_info("random_data")
        print(f"  - get_dataset_info: Successfully retrieved info for random_data")
        
        # Create a simple visualization
        result = create_visualization(
            dataset_name="random_data",
            plot_type="histogram",
            x="numeric_normal",
            filename="test_histogram"
        )
        print(f"  - create_visualization: {result}")
        
        print("  ✓ Data tools test passed\n")
    except Exception as e:
        print(f"  ✗ Data tools test failed: {str(e)}")
        return False
    
    # Test 4: Initialize agent
    print("Test 4: Initializing agent...")
    try:
        agent = create_data_analysis_agent(verbose=False)
        print("  ✓ Successfully initialized agent\n")
    except Exception as e:
        print(f"  ✗ Agent initialization failed: {str(e)}")
        return False
    
    # Test 5: Simple agent query
    print("Test 5: Testing agent with a simple query...")
    try:
        response = agent.invoke({
            "input": "What datasets are currently loaded?"
        })
        print(f"  - Agent response: {response['output'][:100]}...")
        print("  ✓ Agent query test passed\n")
    except Exception as e:
        print(f"  ✗ Agent query test failed: {str(e)}")
        return False
    
    # All tests passed
    print("All tests passed! Your Data Analysis Agent is ready to use.")
    print("\nYou can now run:")
    print("  - python main.py       # For interactive mode")
    print("  - python example.py    # To see example usage")
    
    return True

if __name__ == "__main__":
    run_tests()
