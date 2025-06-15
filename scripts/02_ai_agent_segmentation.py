from io import StringIO

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
from marketing_analysis_team.agents.utils import run_segment_analysis, get_last_human_message

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
You are an expert in marketing analytics for Business Science.

Your task is to analyze whether the user's question requires segmentation analysis and, if so, to:
- Assign meaningful descriptive labels to each segment (segments are identified by numeric IDs like 0, 1, 2, ...).
- Identify key patterns and actionable marketing insights based on segment statistics.
- Return all results in a strict JSON format.

Metrics for each segment include:
- avg_p1: Lead score (0 to 1)
- avg_member_rating: Engagement score (1 to 5)
- avg_purchase_frequency: Average number of transactions per customer
- customer_count: Number of customers in the segment

Make your reasoning precise and business-oriented.

RETURN FORMAT (JSON only — not inside code blocks):

Rules:
- Escape all line breaks as \\n
- Use only double curly braces ({{ and }}) to escape literal curly braces
- Do not include formatting instructions in the output
- No trailing commas or string concatenation
- JSON must be valid and parsable

If analysis is required, return something like:

{{ 
  "general_response": "A brief natural language sentence summarizing the key findings from the customer segmentation.",
  "analysis_required": true,
  "segment_labels": {{
    "0": "Label for segment 0",
    "1": "Label for segment 1",
    "2": "Label for segment 2"
    // Additional segments as needed
  }},
  "insights": "Key insights describing patterns and implications across all segments.",
  "summary_table": "segment_name | avg_p1 | avg_member_rating | avg_purchase_frequency | customer_count\\nLabel 0 | 0.85 | 4.3 | 2.7 | 1300\\nLabel 1 | 0.67 | 4.0 | 2.3 | 1100\\nLabel 2 | 0.45 | 3.2 | 1.9 | 900"
}}

If no analysis is required, return:

{{ 
  "general_response": "No segment analysis needed.",
  "analysis_required": false,
  "segment_labels": {{}},
  "insights": "",
  "summary_table": ""
}}
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
run_segment_analysis(segment_analyzer, test_input, show_raw=True, show_json=True)
''' 
result = segment_analyzer.invoke(test_input)
print("🟡 Raw response from model:")
print(result)

console = Console()
console.rule("[bold green]Segment Analysis Output")
console.print_json(data=result)


# Format the table
raw_table = result["summary_table"].replace("\\n", "\n")
df = pd.read_csv(StringIO(raw_table), sep="|")
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Replace label names
segment_labels = result["segment_labels"]
df["segment_name"] = df["segment_name"].apply(lambda x: segment_labels.get(x.split()[-1], x))

# Build the Markdown string
markdown_report = f"""
# 🧠 Segment Analysis Report

**General Response**  
{result["general_response"]}

**Analysis Required**  
{result["analysis_required"]}

**Segment Labels**  
""" + "\n".join([f'- **{k}**: {v}' for k, v in segment_labels.items()]) + """

**Insights**  
""" + result["insights"] + """

**Summary Table**  
""" + df.to_markdown(index=False)

# Print 
console = Console()
console.print(Markdown(markdown_report))

'''