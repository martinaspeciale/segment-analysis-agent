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
