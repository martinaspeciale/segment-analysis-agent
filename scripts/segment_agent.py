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
        prompt = payload["prompt"]  # ← this is now passed in directly

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
1. **Assign clear, non-generic names** to each segment ID (e.g., “Loyal Advocates”, not “Segment 0”) using the segment_label_keys dictionary.
2. **Write 3–5 unique and data-driven insights** that compare the segments using only the provided statistics.
3. **Only reference segments that appear in the input data**. Do not invent new segment IDs or names.
4. Focus on differences in behavior, potential, and strategic opportunities between segments.
5. When assigning labels, base your choice strictly on the behavior reflected in the statistics. Do not infer psychological or economic motivations (e.g., "price-sensitive") unless they are strongly supported by the data.
6. If a segment shows high engagement and high lead score but low purchases, label them as "unconverted" or "high-intent".
7. If a segment has low values across the board, label them as "inactive", "low-value", or "dormant".
8. Do not make up or adjust any statistics — use the numbers exactly as provided.
9. Do not fabricate a table — this will be handled separately.


Respond with a valid JSON object containing:

{{
  "general_response": "High-level summary of the key findings and patterns.",
  "analysis_required": true,
  "segment_labels": {segment_label_keys},
  "insights": "List of specific insights derived from the segment data.",
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

    df_transactions = pd.read_sql("SELECT user_email, purchased_at FROM transactions", conn)
    conn.close()

    purchase_freq = df_transactions.groupby("user_email").size().reset_index(name="purchase_frequency")
    df_analysis = df_leads.merge(purchase_freq, on="user_email", how="left")
    df_analysis["purchase_frequency"] = df_analysis["purchase_frequency"].fillna(0)

    df_summary = df_analysis.drop_duplicates(subset="user_email").groupby("segment").agg({
        "p1": "mean",
        "member_rating": "mean",
        "purchase_frequency": "mean",
        "user_email": "nunique" # "count" inflated the counts, with "nunique" we are sure to include only unique users per each segment
    }).rename(columns={"user_email": "customer_count"}).reset_index()

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

    segment_label_keys = json.dumps({str(s): f"Label for segment {s}" for s in segments}, indent=2)
    '''
    Tested Models:
     ---> deepseek/deepseek-r1:free
     ---> mistralai/mistral-7b-instruct
    '''
    # llm = ChatOpenRouter(model="deepseek/deepseek-r1:free")
    llm = ChatOpenRouter(model="mistralai/mistral-7b-instruct")
    prompt = segment_analysis_prompt.format_prompt(
        initial_question=last_question,
        chat_history=messages,
        segment_statistics=json.dumps(segment_stats_json),
        segment_label_keys=segment_label_keys
    ).to_string()

    result = llm.invoke({"prompt": prompt})
    default_labels = {str(i): f"Segment {i}" for i in df_summary["segment"]}
    segment_labels = {str(k): v for k, v in result.get("segment_labels", default_labels).items()}
    df_summary["segment_name"] = df_summary["segment"].astype(str).map(segment_labels)

    # Extract known segment names
    known_segment_names = set(df_summary["segment_name"])

    insights = result["insights"]
    if isinstance(insights, list):
        insights = "\n".join(f"- {item}" for item in insights)

    # Filter hallucinated segment names
    insight_lines = insights.strip().split("\n")
    filtered_insights = []
    for line in insight_lines:
        if any(name in line for name in known_segment_names):
            filtered_insights.append(line)
        else:
            print(f"⚠️ Skipping hallucinated insight: {line}")

    # Rebuild cleaned insight block
    cleaned_insights = "\n".join(filtered_insights)

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
        "response": [AIMessage(content=result["general_response"] + "\n\n" + cleaned_insights)],
        "name": "SegmentAnalysisAgent",
        "insights": insights,
        # "summary_table": result.get("summary_table", ""),
        "analysis_required": result.get("analysis_required", False),
        "segment_labels": segment_labels,
        "segmentation_data": df_summary.to_dict(orient="records"),
        "chart_json": chart_json
    }

def strategy_generator_node(state: GraphState) -> GraphState:
    segment_data = state["segmentation_data"]
    label_to_stats = {
        row["segment_name"]: {
            "avg_p1": row["avg_p1"],
            "avg_member_rating": row["avg_member_rating"],
            "avg_purchase_frequency": row["avg_purchase_frequency"],
            "customer_count": row["customer_count"]
        } for row in segment_data
    }
    prompt = f"""
You are a senior marketing strategist.

You are given a set of customer segments, each with a name and associated statistics:
{json.dumps(label_to_stats, indent=2)}

Write one actionable marketing recommendation per segment.
Each should be one sentence, customized to the segment's size, behavior, and value.

Return ONLY a JSON object with this structure:

{{
  "strategy_recommendations": {{
    "Segment Label 1": "Recommendation",
    "Segment Label 2": "Recommendation"
  }}
}}
""".strip()

    llm = ChatOpenRouter(model="mistralai/mistral-7b-instruct")
    raw_response = llm.invoke({"prompt": prompt})

    strategy_recs = {}
    if isinstance(raw_response, dict) and "strategy_recommendations" in raw_response:
        strategy_recs = raw_response["strategy_recommendations"]
    elif isinstance(raw_response, str):
        try:
            strategy_recs = json.loads(raw_response.replace("'", '"')).get("strategy_recommendations", {})
        except Exception:
            matches = re.findall(r'-\s*(.+?):\s*(.+)', raw_response)
            strategy_recs = {name.strip(): rec.strip() for name, rec in matches}

    return {
        **state,
        "strategy_recommendations": strategy_recs,
        "response": state["response"] + [
            AIMessage(content="\n📌 Strategy Recommendations:\n" + "\n".join(f"- {k}: {v}" for k, v in strategy_recs.items()))
        ]
    }

workflow = StateGraph(GraphState)
workflow.add_node("segment_analyzer", segment_analysis_node)

workflow.add_node("strategy_generator", strategy_generator_node)
workflow.set_entry_point("segment_analyzer")
workflow.add_edge("segment_analyzer", "strategy_generator")
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

    df_summary = pd.DataFrame(result["segmentation_data"])
    df_summary = df_summary[["segment_name", "avg_p1", "avg_member_rating", "avg_purchase_frequency", "customer_count"]]

    print("\n📊 Segment Summary Table:")
    print(tabulate(df_summary, headers="keys", tablefmt="github", showindex=False))

    strategy_text = ""

    for msg in result.get("response", []):
        if isinstance(msg, AIMessage) and "Strategy Recommendations" in msg.content:
            strategy_text = msg.content
            break

    if strategy_text:
        print(strategy_text)
    else:
        print("⚠️ No strategy recommendations found in response.")



    # print("\n🧪 Segment IDs returned:")
    # print(sorted(set([row["segment"] for row in result["segmentation_data"]])))
 
'''
in the "Respond with:" of the segment_analysis_prompt we removed:
  "summary_table": (
    "A markdown-style table that summarizes the provided segment statistics. "
    "Each row must correspond to one of the JSON entries above. "
    "Use the exact numeric values from the input JSON — do not round, interpolate, or invent any numbers. "
    "Example format:\n"
    "segment_name | avg_p1 | avg_member_rating | avg_purchase_frequency | customer_count\n"
    "Label 1      | 0.61   | 3.5               | 2.1                    | 1500"
    )
'''