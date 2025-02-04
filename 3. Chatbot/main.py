import streamlit as st
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

# Define mathematical tools
@tool
def add(a: int, b: int) -> str:
    """Adds a and b."""
    return f"{a} + {b} = {a + b}"

@tool
def multiply(a: int, b: int) -> str:
    """Multiplies a and b."""
    return f"{a} × {b} = {a * b}"

@tool
def subtract(a: int, b: int) -> str:
    """Subtracts b from a."""
    return f"{a} - {b} = {a - b}"

@tool
def divide(a: int, b: int) -> str:
    """Divides a by b (if b ≠ 0)."""
    return "Division by zero is not allowed." if b == 0 else f"{a} ÷ {b} = {a / b}"

@tool
def modulus(a: int, b: int) -> str:
    """Finds modulus (remainder) of a divided by b."""
    return f"{a} % {b} = {a % b}"

@tool
def exponentiate(a: int, b: int) -> str:
    """Raises a to the power of b."""
    return f"{a}^{b} = {a ** b}"

@tool
def square_root(a: int) -> str:
    """Finds the square root of a."""
    return f"√{a} = {a ** 0.5}"

# List of tools
tools = [add, multiply, subtract, divide, modulus, exponentiate, square_root]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Function to process chatbot query
def chatbot(query):
    messages = [
        SystemMessage("You are a math assistant. Use available tools for calculations."),
        HumanMessage(query),
    ]

    try:
        ai_msg = llm_with_tools.invoke(messages)

        # Extract tool call responses
        tool_responses = []
        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # Select and invoke the tool
                selected_tool = {
                    "add": add, "multiply": multiply, "subtract": subtract,
                    "divide": divide, "modulus": modulus, "exponentiate": exponentiate,
                    "square_root": square_root
                }.get(tool_name)

                if selected_tool:
                    tool_responses.append(selected_tool.run(tool_args))

        # Return only the tool response
        return tool_responses if tool_responses else ["I couldn't process the request."]

    except Exception as e:
        return [f"Error: {str(e)}"]

# Streamlit UI
st.title("Math Chatbot with Streamlit")
st.write("Ask a math-related question!")

query = st.text_input("Enter your query:")

if query:
    tool_responses = chatbot(query)

    # Display only the computed answer
    for response in tool_responses:
        st.write(response)


##streamlit run "D:\A UDEMY\__Internship__\Assignements\3. Chatbot\main.py"
