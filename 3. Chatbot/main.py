import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM and history
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
history = ChatMessageHistory()

# Function to save conversation history
def save_conversation(user_input, ai_response):
    history.add_user_message(user_input)
    history.add_ai_message(ai_response)

# Streamlit interface
st.title("LlamaAI - Chatbot with Runnable Message History")

# Store messages in session state
if 'history_display' not in st.session_state:
    st.session_state.history_display = []

# Input from the user
user_input = st.text_input("You:")

if user_input:
    # Generate AI response
    response = llm.invoke([("human", user_input)])
    save_conversation(user_input, response.content)

    # Add to history for display
    st.session_state.history_display.append(("You", user_input))
    st.session_state.history_display.append(("AI", response.content))

    # Display the response
    st.write(f"AI: {response.content}")

# Display conversation history with runnable options
st.write("### Conversation History")
for idx, (role, message) in enumerate(st.session_state.history_display):
    st.write(f"*{role}:* {message}")

    # Add a button to rerun the message
    if role == "You":
        if st.button(f"Rerun: {message}", key=f"rerun_{idx}"):
            # Use the message to generate a new response
            response = llm.invoke([("human", message)])
            st.write(f"AI (rerun): {response.content}")
            save_conversation(message, response.content)

# Option to download the conversation
def download_conversation():
    conversation_text = ""
    for message in history.messages:
        role = "You" if message.type == "human" else "AI"
        conversation_text += f"{role}: {message.content}\n\n"
    return conversation_text

conversation_text = download_conversation()
st.download_button("Download Conversation", conversation_text, file_name="conversation.txt", mime="text/plain")


##streamlit run "D:\A UDEMY\__Internship__\Assignements\3. Chatbot\main.py"