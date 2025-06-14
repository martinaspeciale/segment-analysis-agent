## Marketing Analytics Team in a Box 
### 1st stop: Segment-Analysis Agent 

## 🧠 Local AI Agent with Ollama

We integrate a **local language model agent** powered by [Ollama](https://ollama.com), using the `mistral` model as the core reasoning engine.

### 🔧 Steps

- **Install Ollama** 
- Pull and launch the **Mistral model** locally via `ollama pull llama2:7b`.
- Build a LangChain-compatible agent with:
  - `ChatOllama` to interface with the local LLM.
  - Prompts and message parsing using LangChain core components.
  - Connection to a **SQLite database** of scored leads using SQLAlchemy.

### 📂 File Overview

- `scripts/02_ai_agent_segmentation.py`: main script that connects the LLM to lead segmentation logic.
- `data/leads_scored_segmentation.db`: local database used to store and analyze scored leads.

### 🚀 Run the Agent

1. Start the Ollama server in a terminal:
```bash 
ollama serve
```

2. In another terminal, run the agent script:
```bash
python3 scripts/02_ai_agent_segmentation.py
```
The agent will connect to the local model and output a response.
