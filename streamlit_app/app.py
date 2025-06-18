import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from langchain_core.messages import HumanMessage
import sys
import os
from sqlalchemy import text
from plotly.io import from_json
import re
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from segment_agent import get_segment_app
segment_app = get_segment_app()


# Add the ../scripts folder to sys.path
current_dir = os.path.dirname(__file__)
scripts_path = os.path.abspath(os.path.join(current_dir, "..", "scripts"))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

st.set_page_config(page_title="Segment Analysis Agent", layout="wide")
st.title("🧠 Segment Analysis Agent")

# Sidebar: Select Database
st.sidebar.header("🗂 Select Dataset")
available_dbs = [f for f in os.listdir("data") if f.endswith(".db")]
selected_db = st.sidebar.selectbox("Choose a database", available_dbs)

# Build full path and create SQLAlchemy engine
DB_PATH = f"sqlite:///data/{selected_db}"
engine = create_engine(DB_PATH)

# Sidebar: Select Table to Preview
st.sidebar.header("📂 Explore Tables")
with engine.connect() as conn:
    table_names = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()
    table_names = [t[0] for t in table_names]

selected_table = st.sidebar.selectbox("Select a table to preview", table_names)

if selected_table:
    with engine.connect() as conn:
        df_preview = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT 10", conn)
    st.sidebar.subheader(f"🧾 Preview of `{selected_table}`")
    st.sidebar.dataframe(df_preview)

# debug 
st.sidebar.write(f"📁 Using database: {selected_db}")
# Extract expected segment count from filename
match = re.search(r"case(\d+)", selected_db)
case_segment_map = {
    "1": 3,
    "2": 4,
    "3": 3,
    "4": 4,
    "5": 3
}
expected_segments = case_segment_map.get(match.group(1), "unknown") if match else "unknown"
st.sidebar.write(f"🎯 Expected number of segments: {expected_segments}")
## debug end 

st.markdown("---")

# Main panel: Segment Agent
st.header("🔍 Ask the Segment Analysis Agent")
user_question = st.text_area("What would you like to know about your customer segments?", value="Can you analyze the segments and provide insights?", height=100)

if st.button("Run Analysis"):
    with st.spinner("Running segment analysis..."):
        messages = [HumanMessage(content=user_question)]
        # results = segment_app.invoke({"messages": messages})
        db_path = Path("data") / selected_db
        db_url = f"sqlite:///{db_path.resolve()}"

        results = segment_app.invoke({
            "messages": messages,
            "db_path": db_url
        })

        segment_ids = sorted(set([row["segment"] for row in results["segmentation_data"]]))
        st.info(f"🧪 Segments found in DB: {segment_ids}")

        st.subheader("🧠 General Insights")
        st.markdown(results["response"][0].content)
        if len(results["response"]) > 1:
            strategy_msg = results["response"][1].content
            st.subheader("📌 Strategy Recommendations")
            st.markdown(strategy_msg)


        st.subheader("📊 Segment Summary Table")
        summary_df = pd.DataFrame.from_dict(results["segmentation_data"])
        summary_df = summary_df[["segment_name", "avg_p1", "avg_member_rating", "avg_purchase_frequency", "customer_count"]]
        st.dataframe(summary_df)

        st.subheader("📈 Segment Chart")
        fig = from_json(results["chart_json"])
        st.plotly_chart(fig, use_container_width=True)