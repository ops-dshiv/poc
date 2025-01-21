import streamlit as st
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableConfig

# Initialize LangGraph
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from meta3 import compiled_graph  # Replace with the actual import for your compiled graph

import streamlit as st
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableConfig

def pretty_print_stream_chunk(chunk):
    """
    Pretty print updates from each node in the graph execution.
    """
    for node, updates in chunk.items():
        print(f"=== Update from Node: {node} ===")

        if "messages" in updates:
            print("Messages:")
            for message in updates["messages"]:
                print(f"  - Role: {getattr(message, 'role', 'unknown')}")
                print(f"    Content: {getattr(message, 'content', 'N/A')}")

        if "state" in updates:
            print("State Updates:")
            pprint.pprint(updates["state"], indent=4)

        for key, value in updates.items():
            if key not in {"messages", "state"}:
                print(f"Other Update - {key}: {value}")
        print("\n")



# Function to process input and generate response
def process_input(user_input):
    input_state = {
        "messages": [HumanMessage(content=user_input)],
        "memory": st.session_state.get("memory", {"session_context": []}),
    }
    
    config = RunnableConfig(configurable={"user_id": "10", "thread_id": "2"})
    output_chunks = []

    for chunk in compiled_graph.stream(input_state, config=config):
        output_chunks.append(chunk)
        pretty_print_stream_chunk(chunk)  # For debugging, optional

    # Update session memory from output chunks
    for chunk in output_chunks:
        if "state" in chunk.get("generate_response", {}):
            st.session_state.memory = chunk["generate_response"]["state"].get("memory", {})
        
    # Retrieve final response
    final_chunk = output_chunks[-1] if output_chunks else {}
    messages = final_chunk.get("generate_response", {}).get("messages", [])
    if messages:
        return messages[-1].content
    return "No response generated."

# Streamlit App UI
st.title("AI Chatbot with Memory")
st.write("Ask your questions, and the bot will respond based on the workflow logic.")

# Initialize session memory
if "memory" not in st.session_state:
    st.session_state.memory = {"session_context": []}

# Input and Output UI
user_input = st.text_input("Your question:")
if st.button("Submit") and user_input.strip():
    response = process_input(user_input)
    st.write("**Response:**")
    st.write(response)
else:
    st.write("Please enter a question to get started.")



