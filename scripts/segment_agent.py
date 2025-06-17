import os
import json
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from typing import Sequence, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import requests
import plotly.express as px
from io import StringIO
from tabulate import tabulate


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class ChatOpenRouter:
    def __init__(self, model: str):
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

    def invoke(self, payload: dict):
        prompt = segment_analysis_prompt.format(**payload)
        response = requests.post(self.api_url, headers=self.headers, json={
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        })
        response.raise_for_status()
        output = response.json()["choices"][0]["message"]["content"]
        return JsonOutputParser().invoke(output)

segment_analysis_prompt = PromptTemplate(
    input_variables=["initial_question", "chat_history", "segment_statistics", "segment_label_keys"],
    template="""
You are a senior marketing analyst.

You are provided:
- A customer question.
- Chat history.
- Segment statistics as a JSON list, where each item contains:
  - segment (int): ID
  - avg_p1 (float): lead score (0–1)
  - avg_member_rating (float): engagement score (1–5)
  - avg_purchase_frequency (float): avg transactions per customer
  - customer_count (int)

Your tasks:
1. **Always analyze the segment data**.
2. Assign clear, non-generic labels to each segment ID (e.g., “First-Time Explorers”, not “Segment 0”).
3. Write **unique and data-driven insights** based on real differences.
4. Build a **well-formatted markdown table** of the segment statistics, replacing numeric IDs with labels.
5. Return only a valid **JSON** object with the fields below. Do not wrap in code blocks or Markdown.
6. Only refer to segments that appear in the input data. Do not invent or assume any additional segments.

Respond with:
{{
  "general_response": "Natural language summary of key findings.",
  "analysis_required": true,
  "segment_labels": {{segment_label_keys}},
  "insights": "3 to 5 unique insights derived from the segment data. Be specific.",
  "summary_table": "segment_name | avg_p1 | avg_member_rating | avg_purchase_frequency | customer_count\\nLabel 1 | 0.6 | 3.5 | 2.1 | 1500"
}}
"""
)

class GraphState(TypedDict):
    messages: Sequence[BaseMessage]
    response: Sequence[BaseMessage]
    db_path: str  
    insights: str
    summary_table: str
    analysis_required: bool
    segment_labels: dict
    segmentation_data: dict
    chart_json: str

def get_last_human_message(messages):
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m
    return None

def segment_analysis_node(state: GraphState) -> GraphState:
    # engine = create_engine("sqlite:///data/leads_scored_segmentation.db")
    db_path = state.get("db_path", "sqlite:///data/leads_scored_segmentation.db")
    print(f"📂 Loading database: {db_path}")
    engine = create_engine(db_path)
    conn = engine.connect()

    df_leads = pd.read_sql("SELECT user_email, p1, member_rating, segment FROM leads_scored", conn)
    df_leads = df_leads.drop_duplicates(subset="user_email")  # <-- ENSURE unique users

    print("🧪 Raw leads loaded:", len(df_leads))
    print("👤 Unique users in leads:", df_leads['user_email'].nunique())

    df_transactions = pd.read_sql("SELECT user_email, purchased_at FROM transactions", conn)
    conn.close()

    print("🛒 Transactions loaded:", len(df_transactions))
    print("👤 Unique users in transactions:", df_transactions['user_email'].nunique())

    purchase_freq = df_transactions.groupby("user_email").size().reset_index(name="purchase_frequency")
    df_analysis = df_leads.merge(purchase_freq, on="user_email", how="left")
    df_analysis["purchase_frequency"] = df_analysis["purchase_frequency"].fillna(0)

    df_summary = df_analysis.drop_duplicates(subset="user_email").groupby("segment").agg({
        "p1": "mean",
        "member_rating": "mean",
        "purchase_frequency": "mean",
        "user_email": "nunique" # "count" inflated the counts, with "nunique" we are sure to include only unique users per each segment
    }).rename(columns={"user_email": "customer_count"}).reset_index()

    print("✅ Final rows after drop_duplicates:", len(df_analysis))

    df_summary["avg_p1"] = df_summary["p1"].round(3)
    df_summary["avg_member_rating"] = df_summary["member_rating"].round(2)
    df_summary["avg_purchase_frequency"] = df_summary["purchase_frequency"].round(2)
    segments = df_summary["segment"].unique()
    segment_label_keys = "{\n" + ",\n".join([f'"{int(s)}": "Label for segment {int(s)}"' for s in segments]) + "\n}"
    print("🧪 Segment IDs in DB:", df_summary["segment"].unique())

    
    segment_stats_json = df_summary[["segment", "avg_p1", "avg_member_rating", "avg_purchase_frequency", "customer_count"]].to_dict(orient="records")

    messages = state.get("messages")
    last_question = get_last_human_message(messages)
    last_question = last_question.content if last_question else ""

    llm = ChatOpenRouter(model="deepseek/deepseek-r1:free")
    result = llm.invoke({
        "initial_question": last_question,
        "chat_history": messages,
        "segment_statistics": json.dumps(segment_stats_json),
        "segment_label_keys": segment_label_keys
    })

    insights = result["insights"]
    if isinstance(insights, list):
        insights = "\n".join(f"- {item}" for item in insights)

    default_labels = {str(i): f"Segment {i}" for i in df_summary["segment"]}
    segment_labels = {str(k): v for k, v in result.get("segment_labels", default_labels).items()}
    df_summary["segment_name"] = df_summary["segment"].astype(str).map(segment_labels)

    df_viz = df_summary.melt(
        id_vars=["segment_name"],
        value_vars=["avg_p1", "avg_member_rating", "avg_purchase_frequency"],
        var_name="metric",
        value_name="value"
    )

    fig = px.bar(
        df_viz,
        x="segment_name",
        y="value",
        color="metric",
        barmode="group",
        title="Segment Analysis: Lead Score, Member Rating, and Purchase Frequency"
    )

    chart_json = fig.to_json()

    return {
        "response": [AIMessage(content=result["general_response"] + "\n\n" + insights)],
        "name": "SegmentAnalysisAgent",
        "insights": insights,
        "summary_table": result.get("summary_table", ""),
        "analysis_required": result.get("analysis_required", False),
        "segment_labels": segment_labels,
        "segmentation_data": df_summary.to_dict(orient="records"),
        "chart_json": chart_json
    }

workflow = StateGraph(GraphState)
workflow.add_node("segment_analyzer", segment_analysis_node)
workflow.set_entry_point("segment_analyzer")
workflow.add_edge("segment_analyzer", END)
def get_segment_app():
    return workflow.compile()

def run_segment_analysis(user_question: str):
    messages = [HumanMessage(content=user_question)]
    return app.invoke({"messages": messages})


if __name__ == "__main__":
    app = get_segment_app()

    # Explicitly define DBs and their order
    ordered_dbs = [
        ("leads_seg_case1_high_low_engagement.db", "Case 1: High vs. Low Engagement (3 segments)"),
        ("leads_seg_case2_price_vs_loyalty.db", "Case 2: Price Sensitivity vs. Loyalty (4 segments)"),
        ("leads_seg_case3_cluster_outlier.db", "Case 3: Cluster + Outlier (3 segments)"),
        ("leads_seg_case4_narrow_middle.db", "Case 4: Centralized Mass with Margins (4 segments)"),
        ("leads_seg_case5_conversion_ready.db", "Case 5: Conversion Readiness (3 segments)")
    ]

    print("\n🧠 Available Test Databases:\n")
    for i, (_, description) in enumerate(ordered_dbs):
        print(f"{i + 1}. {description}")

    selection = input("\n👉 Select a database by number (1–5): ")
    try:
        selected_index = int(selection) - 1
        selected_file, selected_description = ordered_dbs[selected_index]
    except (ValueError, IndexError):
        print("❌ Invalid selection.")
        exit()

    db_path = f"sqlite:///{os.path.abspath(os.path.join('data', selected_file))}"
    print(f"\n📂 Using DB: {selected_file}")

    # Set the test question
    user_question = "Please analyze the segments and suggest actions."

    # Run the agent
    result = app.invoke({
        "messages": [HumanMessage(content=user_question)],
        "db_path": db_path
    })

    print("✅ LLM Output:")
    print(result["response"][0].content)
    # Usa i dati strutturati veri restituiti dal backend, non quelli dell’LLM
    df_summary = pd.DataFrame(result["segmentation_data"])
    df_summary = df_summary[["segment_name", "avg_p1", "avg_member_rating", "avg_purchase_frequency", "customer_count"]]

    print("\n📊 Segment Summary Table (Real Data):")
    print(tabulate(df_summary, headers="keys", tablefmt="github", showindex=False))


    # print("\n🧪 Segment IDs returned:")
    # print(sorted(set([row["segment"] for row in result["segmentation_data"]])))
 
