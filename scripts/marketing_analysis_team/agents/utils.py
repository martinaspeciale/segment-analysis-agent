from langchain_core.messages import HumanMessage # type: ignore

# Helper function to get last question that the human asked
def get_last_human_message(msgs):
    # Iterate through the list in reverse order 
    for msg in reversed(msgs):
        if isinstance(msg, HumanMessage):
            return msg
    return None