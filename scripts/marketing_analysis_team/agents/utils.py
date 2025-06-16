from langchain_core.messages import HumanMessage 
from rich.console import Console
from rich.markdown import Markdown
import time
import yaml
import requests
from io import StringIO
import pandas as pd
import psutil
import sys


def check_memory(required_gb):
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"🔍 Available RAM: {available_gb:.2f} GB")
    if available_gb < required_gb:
        print(f"❌ Not enough RAM. At least {required_gb} GB required.")
        sys.exit(1)
    print("✅ Enough RAM available.")

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


# Helper function to run and display segment analysis
def run_segment_analysis(pipeline, test_input, show_raw=False, show_json=False):
    console = Console()
    
    # Run the agent pipeline
    result = pipeline.invoke(test_input)
    
    # Optionally show raw model response
    if show_raw:
        print("🟡 Raw response from model:")
        print(result)
    
    # JSON output
    if show_json:
        console.rule("[bold green]Segment Analysis Output")
        console.print_json(data=result)
    
    # Check if analysis is required
    if not result.get("analysis_required"):
        console.print("[yellow]⚠️ No analysis was required according to the model.")
        return

    # Format the table safely
    raw_table = result["summary_table"].replace("\\n", "\n")
    try:
        df = pd.read_csv(StringIO(raw_table), sep="|")
        df.columns = df.columns.str.strip()
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    except Exception as e:
        console.print(f"[red]❌ Failed to parse summary table:\n{e}")
        return

    # Replace label names
    segment_labels = result.get("segment_labels", {})
    df["segment_name"] = df["segment_name"].apply(lambda x: segment_labels.get(x.split()[-1], x))

    # Build the markdown report
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

    # Print nicely
    console.print(Markdown(markdown_report))

# Helper function to get last question that the human asked
def get_last_human_message(msgs):
    # Iterate through the list in reverse order 
    for msg in reversed(msgs):
        if isinstance(msg, HumanMessage):
            return msg
    return None

