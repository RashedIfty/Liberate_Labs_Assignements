import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.3, api_key=groq_api_key)

# =============== Define Tools ===============
# These tools allow the Junior Agent to perform different tasks

# Python REPL tool for executing Python code
python_repl = PythonREPLTool()

# Text processing tool for summarization, grammar checking, etc.
def text_processor(text):
    return llm.invoke(f"Process this text: {text}")

tools = [
    Tool(
        name="CodeInterpreter",
        func=python_repl.run,
        description="Use this tool to run Python code for calculations or data analysis."
    ),
    Tool(
        name="TextProcessor",
        func=text_processor,
        description="Use this tool to process and refine text."
    )
]

# =============== Senior Agent (Task Decomposer) ===============
# This agent analyzes the user's input and breaks it down into subtasks

senior_prompt = PromptTemplate(
    input_variables=["input"],
    template=""" 
    SYSTEM: You are a Senior Prompt Engineer. Analyze and break down the task clearly into simple, actionable steps.

    USER QUERY: {input}

    RESPONSE FORMAT:
    1. Subtask 1: [Description of Subtask 1]
    2. Subtask 2: [Description of Subtask 2]
    """
)

# Create a LangChain sequential chain to process tasks
senior_chain = LLMChain(llm=llm, prompt=senior_prompt)

# =============== Junior Agent (Task Executor) ===============
# This agent selects the right tool and executes the subtasks

junior_prompt = PromptTemplate(
    input_variables=["subtask"],
    template=""" 
    SYSTEM: You are a Junior Prompt Engineer. Execute the given subtask:

    SUBTASK: {subtask}

    Select the appropriate tool: [CodeInterpreter | TextProcessor]
    Provide a detailed output with explanations and sources.
    """
)

# Initialize the agent with tools
junior_agent = initialize_agent(
    tools=tools,  # Assigning the tools created above
    llm=llm,  # Using the same model
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Agent type for multi-tool reasoning
    verbose=True  # Enables detailed logging in console
)

# =============== Execution Workflow ===============
# This function processes the query step-by-step

def execute_pipeline(user_input):
    """Runs the Senior & Junior Agents in sequence"""
    
    # Step 1: Senior Agent analyzes the task and breaks it into subtasks
    with st.spinner("🔍 Analyzing the task..."):
        senior_response = senior_chain.run(input=user_input)

    # Display Senior Agent's output (Debugging the response)
    st.subheader("📌 Senior Agent Analysis")
    st.write(senior_response)  # Debugging: Check the full response from Senior Agent

    # Extract subtasks (Simple approach: look for "Subtask" markers)
    subtasks = []
    for line in senior_response.split("\n"):
        if "Subtask" in line:
            # Extract description of each subtask
            subtask = line.split(":")[1].strip() if ":" in line else ""
            if subtask:
                subtasks.append(subtask)

    # Validate that subtasks are not empty
    if not subtasks or any(not subtask for subtask in subtasks):
        st.error("⚠️ The Senior Agent did not generate valid subtasks. Please check the input.")
        return

    # Step 2: Junior Agent executes each subtask
    st.subheader("🛠 Junior Agent Execution")
    final_results = []

    for subtask in subtasks:
        st.write(f"**Executing:** {subtask}")  # Display current subtask

        try:
            with st.spinner(f"⏳ Processing subtask: {subtask}..."):
                # Ensure subtask is valid
                if subtask:
                    junior_response = junior_agent.run(subtask=subtask, handle_parsing_errors=True)

            st.success(f"✅ {subtask} completed!")  # Success message
            st.write(junior_response)  # Display the result
            final_results.append(junior_response)  # Store results

        except Exception as e:
            st.error(f"❌ Error processing subtask: {subtask}\n{str(e)}")

    return "\n\n".join(final_results)


st.title("🚀 AI Prompt Engineer Assistant")
st.markdown("This app **decomposes complex tasks** and **executes them intelligently** using AI agents.")

# User Input
user_query = st.text_area("📝 Enter your task:", placeholder="Example: Compare LLMs like GPT-4 and Claude-3 on benchmark performance")

# Run Agents
if st.button("Run Agents 🚀"):
    if user_query.strip():  # Check if input is valid
        final_output = execute_pipeline(user_query)  # Run the agent pipeline
        st.subheader("📌 Final Output")
        st.write(final_output)
    else:
        st.warning("⚠ Please enter a valid task!")  # Warn if input is empty





#streamlit run "D:\A UDEMY\__Internship__\Assignements\5. Roleplaying agent\main.py"