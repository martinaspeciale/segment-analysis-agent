# AGENTS
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from typing import Sequence, TypedDict
import os
import yaml

# DATA SCIENCE
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import plotly.io as pio

# DEBUGGING
from pprint import pprint
from IPython.display import Markdown, Image, display

# CUSTOM UTILITIES
from marketing_analysis_team.agents.utils import get_last_human_message  # adjust path if needed

# SQLITE DB PATH
db_path = "sqlite:///../data/leads_scored_segmentation.db"

# LANGCHAIN LLM SETUP WITH OLLAMA
model = "tinyllama"  # or any other pulled model (e.g., mistral, llama3, gemma, deepseek)
llm = ChatOllama(model=model)

# TEST THE LLM
response = llm.invoke("What's the recipe for a margarita?")
pprint(response.content)
display(Markdown(response.content))
