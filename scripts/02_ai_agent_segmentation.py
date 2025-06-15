# === TERMINAL RENDERING ===
from rich.console import Console
from rich.markdown import Markdown

# === AGENTS & LANGCHAIN ===
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from typing import Sequence, TypedDict

# === SYSTEM UTILITIES ===
import os
import time
import yaml
import requests

# === DATA SCIENCE STACK ===
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import plotly.io as pio

# === DEBUGGING ===
from pprint import pprint

# === CUSTOM UTILITIES ===
from marketing_analysis_team.agents.utils import get_last_human_message  

# === OLLAMA LOW-LEVEL ===
import ollama


def wait_for_ollama_ready(timeout: int = 60) -> bool:
    """
    Wait for Ollama server to become ready (listening on localhost:11434).
    
    Args:
        timeout (int): Maximum time to wait in seconds.

    Raises:
        TimeoutError: If Ollama doesn't start within the timeout.

    Returns:
        bool: True if Ollama is ready.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get("http://localhost:11434")
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("❌ Ollama did not start in time.")


# === DATABASE CONFIGURATION ===
db_path = "sqlite:///../data/leads_scored_segmentation.db"


# === LLM SETUP ===
model = "gemma:2b"  # Alternative: "llama3", "mistral", etc.
llm = ChatOllama(model=model)

# === WAIT FOR OLLAMA SERVER ===
wait_for_ollama_ready()

'''
# === TEST OLLAMA CHAT ===
response = ollama.chat(
    model=model,
    messages=[
        {'role': 'user', 'content': "What is the recipe for pizza?"}
    ]
)

# === RENDER MARKDOWN OUTPUT TO TERMINAL ===
console = Console()
markdown_text = response['message']['content']
console.print(Markdown(markdown_text))
'''

segment_analysis_prompt = PromptTemplate(
    input_variables=["initial_question", "chat_history", "segment_statistics"],
    template="""
You are an expert in marketing analytics for Business Science,
a premium data science educational platform.
Analyze the user's request to determine if it requires
analyzing customer segments from the database.

The segments are precomputed with numeric IDs (e.g., 0, 1, 2).
Your task is to:

1. Generate descriptive labels for each segment based on their statistics.
2. Provide insights into patterns across segments. Call this section "Segment Insights".
3. Suggest detailed marketing implications and campaign strategies for the different segments based on their attributes. Call this section "Marketing Implications".

Metrics provided:
- avg_p1: Lead score (0 to 1, higher means more likely to purchase).
- avg_member_rating: Engagement rating (1 to 5, higher means more engaged).
- avg_purchase_frequency: Average number of transactions per customer.
- customer_count: Number of customers in the segment.

If segment analysis is requested, provide:
1. A general response summarizing the analysis.
2. A dictionary mapping segment IDs to descriptive labels (e.g., {{ "0": "High-Value Customers" }}).
3. Detailed insights explaining patterns and marketing implications.
4. A summary table of segment statistics in markdown format, using the generated labels.

In the general response, include:
- A summary of the analysis.
- Provide insights into patterns across segments. Call this section "Segment Insights".
- Suggest campaign strategies for the different segments. Title this as "Marketing Implications".
- Use bullets and tables to make the response clear and easy to read.

RETURN FORMAT:
Make sure the response is valid JSON: no trailing commas, no comments, no string concatenation, and escape all line breaks as '\\n'.
A strict JSON object (check to make sure it is valid JSON) with the following:
- If analysis is requested:
{{{{
  "general_response": "Summary of the analysis",
  "analysis_required": true,
  "segment_labels": {{{{
    "0": "Label for segment 0",
    "1": "Label for segment 1",
    "2": "Label for segment 2"
  }}}},
  "insights": "Detailed explanation of patterns and marketing implications.",
  "summary_table": "Provide a single-line Markdown table string, without using + for concatenation. Escape newlines as \\n inside the JSON string."
}}}}

- If no analysis is required:
{{{{
  "general_response": "Response indicating no segment analysis needed",
  "analysis_required": false,
  "segment_labels": {{}},
  "insights": "",
  "summary_table": ""
}}}}
"""
)


# === TEST SEGMENT ANALYSIS AGENT ===

test_input = {
    "initial_question": "Can you analyze the segments and provide insights?",
    "chat_history": [],
    "segment_statistics": [
        {
            "segment": 0,
            "avg_p1": 0.85,
            "avg_member_rating": 4.2,
            "avg_purchase_frequency": 3.1,
            "customer_count": 1500
        },
        {
            "segment": 1,
            "avg_p1": 0.35,
            "avg_member_rating": 2.9,
            "avg_purchase_frequency": 1.2,
            "customer_count": 4200
        },
        {
            "segment": 2,
            "avg_p1": 0.15,
            "avg_member_rating": 1.4,
            "avg_purchase_frequency": 0.3,
            "customer_count": 700
        }
    ]
}
segment_analyzer = segment_analysis_prompt | llm | JsonOutputParser()

result = segment_analyzer.invoke(test_input)
print("🟡 Raw response from model:")
print(result)

console = Console()
console.rule("[bold green]Segment Analysis Output")
console.print_json(data=result)

