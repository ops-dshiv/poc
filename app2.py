# import streamlit as st
# from meta5 import EducationAssistant  # Ensure this module is accessible

# # Initialize AI Assistant
# assistant = EducationAssistant()

# # Streamlit App UI
# st.title("📚 AI Chatbot with Persistent Memory")
# st.write("Ask your questions, and the bot will remember past conversations.")

# # Initialize session state for memory
# if "conversation" not in st.session_state:
#     st.session_state.conversation = []

# # User Input
# user_input = st.text_input("Your question:")

# if st.button("Submit") and user_input.strip():
#     thread_id = "session_1"  # Unique thread ID for session tracking

#     # Prepare input with full history
#     input_state = {
#         "query": user_input,
#         "history": st.session_state.conversation,
#         "thread_id": thread_id
#     }

#     # Invoke AI assistant
#     result = assistant.graph.invoke(input_state)
#     response = result.get("response", "Error processing request")

#     # Store conversation persistently
#     st.session_state.conversation.append(f"User: {user_input}")
#     st.session_state.conversation.append(f"Bot: {response}")

#     # Display response
#     st.write("**Response:**")
#     st.write(response)

# # Show Persistent Conversation History
# st.subheader("📝 Conversation History")
# for message in st.session_state.conversation:
#     st.write(message)


import streamlit as st
from meta5 import EducationAssistant  # Ensure this module is accessible

# Initialize AI Assistant
assistant = EducationAssistant()

# Streamlit App UI
st.title("📚 AI Chatbot with Persistent Memory")
st.write("Ask your questions, and the bot will remember past conversations.")

# Initialize session state for memory
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# User Input
user_input = st.text_input("Your question:")

if st.button("Submit") and user_input.strip():
    thread_id = "session_1"  # Unique thread ID for session tracking

    # Prepare input with full history
    input_state = {
        "query": user_input,
        "history": st.session_state.conversation,
        "thread_id": thread_id
    }

    # Ensure LangGraph gets the required 'checkpoint_ns' and 'checkpoint_id'
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "edu_assistant",
            "checkpoint_id": f"session_{thread_id}"
        }
    }

    try:
        # Invoke AI assistant with proper configuration
        result = assistant.graph.invoke(input_state, config)
        response = result.get("response", "Error processing request")

        # Store conversation persistently
        st.session_state.conversation.append(f"User: {user_input}")
        st.session_state.conversation.append(f"Bot: {response}")

        # Display response
        st.write("**Response:**")
        st.write(response)

    except Exception as e:
        st.write(f"⚠️ Error: {str(e)}")

# Show Persistent Conversation History
st.subheader("📝 Conversation History")
for message in st.session_state.conversation:
    st.write(message)

