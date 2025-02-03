# import streamlit as st
# from meta7 import EducationAssistant

# # Initialize AI Assistant
# assistant = EducationAssistant()

# st.title("📚 AI Chatbot with Persistent Memory")
# st.write("Ask your questions, and the bot will remember past conversations.")

# if "conversation" not in st.session_state:
#     st.session_state.conversation = []

# user_input = st.text_input("Your question:")

# if st.button("Submit") and user_input.strip():
#     thread_id = "session_1"

#     input_state = {
#         "query": user_input,
#         "history": st.session_state.conversation,
#         "thread_id": thread_id
#     }

#     result = assistant.graph.invoke(input_state)
#     response = result.get("response", "Error processing request")

#     st.session_state.conversation.append(f"User: {user_input}")
#     st.session_state.conversation.append(f"Bot: {response}")

#     st.write("**Response:**")
#     st.write(response)

# st.subheader("📝 Conversation History")
# for message in st.session_state.conversation:
#     st.write(message)


import streamlit as st
from meta10 import EducationAssistant

assistant = EducationAssistant()

st.title("📚 AI Chatbot with Memory")
st.write("Ask your questions, and the bot will remember past conversations.")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("Your question:")

if st.button("Submit") and user_input.strip():
    thread_id = "session_1"

    input_state = {"query": user_input, "history": st.session_state.conversation, "thread_id": thread_id}

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "edu_assistant", "checkpoint_id": f"session_{thread_id}"}}

    try:
        result = assistant.graph.invoke(input_state, config)
        response = result.get("response", "Error processing request")

        st.session_state.conversation.append(f"User: {user_input}")
        st.session_state.conversation.append(f"Bot: {response}")

        st.write("**Response:**")
        st.write(response)

    except Exception as e:
        st.write(f"⚠️ Error: {str(e)}")

st.subheader("📝 Conversation History")
for message in st.session_state.conversation:
    st.write(message)
