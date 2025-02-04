import os
import requests
from dotenv import load_dotenv
from groq import ChatGroq
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Llama (Groq)
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

# Initialize ChromaDB
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma"))

# Create a ChromaDB collection to store the data
collection = client.create_collection(name="multi_tool_data")

class WebFetcher:
    def fetch_data(self, url):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.text[:500]  # Limit data to prevent overload
                
                # Use LLM to process or summarize the fetched data
                prompt = f"Here is some data fetched from a website: {data}. Please summarize it."
                summary_response = llm.chat(messages=[{"role": "user", "content": prompt}])
                summary = summary_response.choices[0].message["content"].strip()

                # Store the fetched data and its summary in ChromaDB
                collection.add(
                    documents=[data, summary],
                    metadatas=[{"source": "web", "url": url}],
                    ids=["web_data"]
                )
                return f"Fetched Data: {data}\nSummary: {summary}"
            return f"Error: Unable to fetch data from {url} (Status Code: {response.status_code})"
        except requests.exceptions.RequestException as e:
            return f"Request failed: {e}"

class TextSummarizer:
    def summarize(self, text):
        try:
            prompt = f"Summarize this text:\n{text}"
            response = llm.chat(messages=[{"role": "user", "content": prompt}])
            summary = response.choices[0].message["content"].strip()
            
            # Store the summary in ChromaDB
            collection.add(
                documents=[summary],
                metadatas=[{"source": "summary"}],
                ids=["summary"]
            )
            return f"Summarized Text: {summary}"
        except Exception as e:
            return f"Error in summarization: {e}"

class SimpleCalculator:
    def calculate(self, num1, operator, num2):
        try:
            expression = f"{num1} {operator} {num2}"
            prompt = f"Calculate the following expression: {expression}"
            response = llm.chat(messages=[{"role": "user", "content": prompt}])
            result = response.choices[0].message["content"].strip()
            
            # Store the result in ChromaDB
            collection.add(
                documents=[result],
                metadatas=[{"source": "calculator", "operation": operator}],
                ids=["calculation_result"]
            )
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

class MultiToolAgent:
    def __init__(self):
        self.web_fetcher = WebFetcher()
        self.text_summarizer = TextSummarizer()
        self.simple_calculator = SimpleCalculator()

    def use_tool(self, tool_name, *args):
        tools = {
            'web_fetcher': self.web_fetcher.fetch_data,
            'text_summarizer': self.text_summarizer.summarize,
            'simple_calculator': self.simple_calculator.calculate
        }
        if tool_name in tools:
            return tools[tool_name](*args)
        return "Tool not recognized"

def main():
    agent = MultiToolAgent()
    print("Welcome to MultiToolAgent! Type 'exit' to quit.")

    while True:
        print("\nChoose a tool:")
        print("1. Web Fetcher (fetch data from a URL and summarize it)")
        print("2. Text Summarizer (summarize a long text)")
        print("3. Simple Calculator (perform basic arithmetic using LLM)")
        tool_choice = input("Enter the tool number (1/2/3): ").strip()

        if tool_choice == '1':
            url = input("Enter a URL to fetch: ").strip()
            print(agent.use_tool('web_fetcher', url))

        elif tool_choice == '2':
            text = input("Enter text to summarize: ").strip()
            print(agent.use_tool('text_summarizer', text))

        elif tool_choice == '3':
            try:
                num1 = float(input("Enter first number: "))
                operator = input("Enter operator (add, subtract, multiply, divide): ").strip().lower()
                num2 = float(input("Enter second number: "))
                print(agent.use_tool('simple_calculator', num1, operator, num2))
            except ValueError:
                print("Invalid input! Please enter numeric values.")

        elif tool_choice.lower() == 'exit':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
