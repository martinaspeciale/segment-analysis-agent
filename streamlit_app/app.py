import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from langchain_core.messages import HumanMessage
import sys
import os
from sqlalchemy import text
from plotly.io import from_json



# Add the ../scripts folder to sys.path
current_dir = os.path.dirname(__file__)
scripts_path = os.path.abspath(os.path.join(current_dir, "..", "scripts"))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from segment_agent import app as segment_app


st.set_page_config(page_title="Segment Analysis Agent", layout="wide")
st.title("🧠 Segment Analysis Agent")

# Database
DB_PATH = "sqlite:///data/leads_scored_segmentation.db"
engine = create_engine(DB_PATH)

# Sidebar: DB Tables
st.sidebar.header("📂 Explore Database")
with engine.connect() as conn:
    table_names = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()
    table_names = [t[0] for t in table_names]

selected_table = st.sidebar.selectbox("Select a table to preview", table_names)

if selected_table:
    with engine.connect() as conn:
        df_preview = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT 10", conn)
    st.sidebar.subheader(f"🧾 Preview of `{selected_table}`")
    st.sidebar.dataframe(df_preview)

st.markdown("---")

# Main panel: Segment Agent
st.header("🔍 Ask the Segment Analysis Agent")
user_question = st.text_area("What would you like to know about your customer segments?", value="Can you analyze the segments and provide insights?", height=100)

if st.button("Run Analysis"):
    with st.spinner("Running segment analysis..."):
        messages = [HumanMessage(content=user_question)]
        results = segment_app.invoke({"messages": messages})

        st.subheader("🧠 General Insights")
        st.markdown(results["response"][0].content)

        st.subheader("📊 Segment Summary Table")
        summary_df = pd.DataFrame.from_dict(results["segmentation_data"])
        summary_df = summary_df[["segment_name", "avg_p1", "avg_member_rating", "avg_purchase_frequency", "customer_count"]]
        st.dataframe(summary_df)

        st.subheader("📈 Segment Chart")
        fig = from_json(results["chart_json"])
        st.plotly_chart(fig, use_container_width=True)