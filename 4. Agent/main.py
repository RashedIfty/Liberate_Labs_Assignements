class WebFetcher:
    def fetch_data(self, url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.text[:500]
            else:
                return f"Error: Unable to fetch data from {url}"
        except requests.exceptions.RequestException as e:
            return f"Request failed: {e}"

class TextSummarizer:
    def summarize(self, text):
        return text[:100] + "..." if len(text) > 100 else text


class SimpleCalculator:
    def calculate(self, num1, operator, num2):
        try:
            if operator == 'add':
                return num1 + num2
            elif operator == 'subtract':
                return num1 - num2
            elif operator == 'multiply':
                return num1 * num2
            elif operator == 'divide':
                if num2 == 0:
                    return "Error: Cannot divide by zero"
                return num1 / num2
            else:
                return "Invalid operator. Use 'add', 'subtract', 'multiply', or 'divide'."
        except Exception as e:
            return f"Error: {e}"


class MultiToolAgent:
    def __init__(self):
        self.web_fetcher = WebFetcher()
        self.text_summarizer = TextSummarizer()
        self.simple_calculator = SimpleCalculator()

    def use_tool(self, tool_name, *args):
        if tool_name == 'web_fetcher':
            return self.web_fetcher.fetch_data(*args)
        elif tool_name == 'text_summarizer':
            return self.text_summarizer.summarize(*args)
        elif tool_name == 'simple_calculator':
            return self.simple_calculator.calculate(*args)
        else:
            return "Tool not recognized"


def main():
    agent = MultiToolAgent()
    
    print("Welcome to the MultiToolAgent! Type 'exit' to quit.")
    
    while True:
        print("\nChoose a tool:")
        print("1. Web Fetcher (e.g., fetch data from a URL)")
        print("2. Text Summarizer (e.g., summarize a long text)")
        print("3. Simple Calculator (e.g., perform basic arithmetic operations)")
        tool_choice = input("Enter the tool number (1/2/3): ")

        if tool_choice == '1':
            url = input("Enter a URL to fetch: ")
            result = agent.use_tool('web_fetcher', url)
            print(f"Fetched Data: {result}")
        
        elif tool_choice == '2':
            text = input("Enter the text to summarize: ")
            result = agent.use_tool('text_summarizer', text)
            print(f"Summarized Text: {result}")
        
        elif tool_choice == '3':
            num1 = float(input("Enter the first number: "))
            operator = input("Enter the operator (add, subtract, multiply, divide): ").strip().lower()
            num2 = float(input("Enter the second number: "))
            result = agent.use_tool('simple_calculator', num1, operator, num2)
            print(f"Result: {result}")
        
        elif tool_choice.lower() == 'exit':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
