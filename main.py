"""
Data Analysis Agent using LangChain, served via Flask.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import re # For cleaning <think> tags
import ast # For safely evaluating stringified dicts

from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish

from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

from data_tools import (
    load_csv, load_excel, list_datasets, get_dataset_info,
    query_data, describe_column, create_visualization,
    create_interactive_visualization, run_analysis, generate_sample_data
)

load_dotenv()

class ThoughtCapturingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.steps: List[str] = []
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        thought = action.log.strip()
        if thought: self.steps.append(f"Thought:\n{thought}")
        self.steps.append(f"Action: Using tool '{action.tool}' with input '{action.tool_input}'")
    def on_tool_end(self, output: str, **kwargs: Any) -> None: # output is usually str(tool_return_value)
        print(f"[DEBUG ThoughtCapturingCallbackHandler] on_tool_end received output: {output!r} (type: {type(output)})")
        self.steps.append(f"Observation: {output}")
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        final_thought = finish.log.strip()
        if final_thought: self.steps.append(f"Final Thought:\n{final_thought}")
    def get_steps(self) -> List[str]: return self.steps
    def reset_steps(self) -> None: self.steps = []

langfuse_handler = None
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    try:
        langfuse_handler = LangfuseCallbackHandler(public_key=os.getenv("LANGFUSE_PUBLIC_KEY"), secret_key=os.getenv("LANGFUSE_SECRET_KEY"), host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
        print("Langfuse callback handler initialized.")
    except Exception as e: print(f"Error initializing Langfuse handler: {e}")
else: print("Langfuse keys not found, skipping Langfuse initialization.")

agent_executor_instance: Optional[AgentExecutor] = None

def create_data_analysis_agent(model_name: Optional[str] = None, temperature: float = 0) -> Union[AgentExecutor, str]:
    if not os.getenv("OLLAMA_BASE_URL"): return "Error: OLLAMA_BASE_URL not set."
    model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3")
    try:
        llm = ChatOllama(model=model_name, temperature=temperature, base_url=os.getenv("OLLAMA_BASE_URL"))
        llm.invoke("Hello") # Test connection
        print(f"Successfully connected to Ollama model: {model_name} at {os.getenv('OLLAMA_BASE_URL')}")
    except Exception as e: return f"Error initializing LLM '{model_name}': {e}"

    tools = [load_csv, load_excel, list_datasets, get_dataset_info, query_data, describe_column, create_visualization, create_interactive_visualization, run_analysis, generate_sample_data]
    system_message = """
    You are a data analysis assistant. Your goal is to help users analyze and visualize data.
    You have access to tools for loading data (CSV, Excel), listing datasets, getting info,
    querying, describing columns, and creating static or interactive visualizations.
    You can also run predefined analyses or generate sample data.
    If a visualization tool returns structured data (like chartData), prioritize that.
    Your final output should be a helpful summary. If a chart was generated and chartData is available from the tool,
    ensure your textual summary complements this and doesn't just repeat the file paths.
    """ # Simplified system message for brevity in this example
    prompt = ChatPromptTemplate.from_messages([("system", system_message), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}"), MessagesPlaceholder(variable_name="agent_scratchpad")])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False, handle_parsing_errors=True)

app = Flask(__name__)
CORS(app, resources={r"/chat*": {"origins": "*"}})

def initialize_agent_and_data():
    global agent_executor_instance
    print("Initializing Data Analysis Agent for Flask app...")
    agent_or_error = create_data_analysis_agent()
    if isinstance(agent_or_error, str):
        print(f"Failed to initialize agent: {agent_or_error}")
        agent_executor_instance = agent_or_error # Store error message
    else:
        agent_executor_instance = agent_or_error
        print("Data Analysis Agent is ready.")
        # Optional: Initial sample data generation (currently commented out)

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    global agent_executor_instance
    if agent_executor_instance is None: return jsonify({"error": "Agent not initialized."}), 500
    if isinstance(agent_executor_instance, str): return jsonify({"error": f"Agent init failed: {agent_executor_instance}"}), 500

    data = request.get_json()
    if not data or 'query' not in data: return jsonify({"error": "Missing 'query'."}), 400

    user_query = data['query']
    thought_capturer = ThoughtCapturingCallbackHandler()
    all_callbacks = [thought_capturer] + ([langfuse_handler] if langfuse_handler else [])

    try:
        response_dict = agent_executor_instance.invoke({"input": user_query}, config={"callbacks": all_callbacks})
        print(f"[DEBUG main.py] Full agent response from invoke: {response_dict}")
        
        raw_agent_final_output_text = response_dict.get('output', "Agent did not provide standard text output.")
        thinking_steps = thought_capturer.get_steps()

        visualization_data = None
        text_for_ui = ""

        # Priority 1: Check if the agent's final 'output' is the structured dict from our tool
        if isinstance(raw_agent_final_output_text, dict) and \
           ('chartData' in raw_agent_final_output_text or 'filePath_html' in raw_agent_final_output_text):
            print("[DEBUG main.py] Agent's final 'output' IS the structured tool result.")
            visualization_data = raw_agent_final_output_text.get('chartData')
            text_for_ui = raw_agent_final_output_text.get('textSummary', "Visualization generated.")
        else:
            # Priority 2: If final 'output' is text, try to find structured data in the last tool observation
            last_tool_output_dict = None
            for step in reversed(thinking_steps):
                if step.startswith("Observation:"):
                    obs_content = step.replace("Observation:", "").strip()
                    try:
                        potential_dict = ast.literal_eval(obs_content) # Safely evaluate string representation of dict
                        if isinstance(potential_dict, dict) and \
                           ('chartData' in potential_dict or 'filePath_html' in potential_dict):
                            last_tool_output_dict = potential_dict
                            print(f"[DEBUG main.py] Parsed structured data from last observation: {last_tool_output_dict}")
                            break
                    except (ValueError, SyntaxError) as e:
                        print(f"[DEBUG main.py] Failed to parse observation as dict: {obs_content}, Error: {e}")
                        pass 
            
            if last_tool_output_dict:
                visualization_data = last_tool_output_dict.get('chartData')
                tool_text_summary = last_tool_output_dict.get('textSummary', "Visualization details from tool observation.")
                
                # Clean the agent's final natural language summary
                llm_summary_cleaned = ""
                if isinstance(raw_agent_final_output_text, str):
                    llm_summary_cleaned = re.sub(r'<think>.*?</think>\s*', '', raw_agent_final_output_text, flags=re.DOTALL).strip()
                
                # Combine tool summary and LLM's summary intelligently
                if llm_summary_cleaned and llm_summary_cleaned != tool_text_summary:
                    text_for_ui = f"{tool_text_summary}\n\nAgent's further comments: {llm_summary_cleaned}"
                else:
                    text_for_ui = tool_text_summary or llm_summary_cleaned # Use whichever is non-empty
            
            elif isinstance(raw_agent_final_output_text, str): # No structured data found, use cleaned final output
                text_for_ui = re.sub(r'<think>.*?</think>\s*', '', raw_agent_final_output_text, flags=re.DOTALL).strip()
                if not text_for_ui and raw_agent_final_output_text.strip(): # If stripping made it empty
                     text_for_ui = raw_agent_final_output_text.replace("<think>", "[Thought:] ").replace("</think>", "")
            else: # Fallback
                text_for_ui = str(raw_agent_final_output_text)

        response_payload = {
            "textResult": text_for_ui or "Agent provided a response.",
            "visualization": visualization_data,
            "thinkingSteps": thinking_steps
        }
        print(f"Flask sending response: {{'textResult': '{str(text_for_ui)[:100]}...', 'visualization': {visualization_data is not None}, 'thinkingSteps': {len(thinking_steps)}}}")
        return jsonify(response_payload)

    except Exception as e:
        print(f"Error during agent invocation: {e}")
        thinking_steps_on_error = thought_capturer.get_steps()
        return jsonify({"error": f"Error processing query: {str(e)}", "thinkingSteps": thinking_steps_on_error}), 500

if __name__ == "__main__":
    initialize_agent_and_data()
    app.run(host='0.0.0.0', port=int(os.getenv("FLASK_PORT", 5001)), debug=False)
