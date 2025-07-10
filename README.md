# Segment Analysis Agent

**An AI-powered customer segmentation and analysis tool using Streamlit, SQLite, and OpenRouter's LLMs.**

---

## Folder Structure

```
segment-analysis-agent/
├── streamlit_app 
│   └── app.py
├── data/                                 --> run scripts/generate_different_segmentations.py to populate it
│   ├── leads_scored_segmentation.db      (master database with all assigned segments)
│   ├── leads_seg_caseX_*.db               (filtered subsets per Case for analysis)
│   └── km_*, gmm_*, hclust_*, gt_*.db    (clustering scenario variations)
├── scripts/
│   ├── generate_different_segmentations.py 
│   ├── segment_agent.py
│   ├── 01_generate_segmentation.py        (legacy - from earlier version)
│   ├── 02_ai_agent_segmentation           (legacy - from earlier version)                              
│   └── generate_leads_and_transactions.py (legacy - from earlier version)  
├── .env                                   --> holds the OPENROUTER_API_KEY
├── requirements.txt
└── (other folders and files as needed)
```


> **Note:** The files inside `data/leads_scored_segmentation.db` and `scripts/` (01_*, 02_*, generate_leads_and_transactions.py) belong to a previous version and are kept for reference, but they are not used in the current version of the project.

---

## Project Description

This tool allows to:

- Generate synthetic customer segmentation datasets, including filtered leads_seg_caseX_*.db required for analysis.
- Perform segment analysis using a large language model (LLM).
- Interactively explore segment insights and strategy recommendations via a Streamlit web app.

It uses OpenRouter models (e.g., **Mistral**) to analyze segment behavior and generate actionable insights.

---

## Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone <repository-url>
cd project_root
```

### 2️⃣ Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

Then install required packages:

```bash
pip install -r requirements.txt
```

### 3️⃣ Create `.env` file

You need a valid OpenRouter API key.

Create a `.env` file in the root folder:

```
OPENROUTER_API_KEY=your_api_key_here
```

Without this, the segment agent won't be able to query the LLM.

### 4️⃣ Generate synthetic databases

Before running the app, populate the `data/` folder with all required datasets:

```bash
python scripts/generate_different_segmentations.py
```

This single command will generate:

- **leads_seg_caseX_*.db**  
  → For **Streamlit** analysis (each Case's filtered leads_scored + transactions tables)

- **km_*.db**, **gmm_*.db**, **hclust_*.db**, **gt_*.db**  
  → Clustering scenario variations for experimentation


## Running the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

### Workflow

- Select one of the generated SQLite databases from the sidebar.
- Choose a table to preview.
- Type a question (or leave the default one).
- Click **Run Analysis** to invoke the Segment Analysis Agent.
- The app will:
  - Extract segment statistics.
  - Call the LLM via OpenRouter.
  - Display:
    - General insights
    - Segment summary table
    - Visualization chart

---

## Components Summary

- **scripts/generate_different_segmentations.py**  
  Generates synthetic customer segmentation datasets for different scenarios to test.

- **scripts/segment_agent.py**  
  Loads data, extracts segment statistics, sends prompts to OpenRouter, parses responses, and generates insights & strategies.

- **streamlit_app/app.py**  
  Provides the full interactive web interface via Streamlit.

---

## Important Notes

- You must have a valid OpenRouter account and API key.
- The generated datasets simulate various segmentation situations.
- The prompting logic is carefully designed to minimize hallucination and enforce structured JSON output.
- The generated datasets include **leads_seg_caseX_*.db files required for the Streamlit app**.


---

## Example Prompt

> **"Analyze the segments and suggest actions to improve customer retention."**

The model will generate segment names, key insights, and marketing strategy recommendations based on the customer data.
