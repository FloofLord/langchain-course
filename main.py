from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return "Tokyo weather is sunny"

# gemma3:270m faster model but is not able to use tools
llm = ChatOllama(temperature=0,model="gemma4:latest")


# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.8-flash",
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )


tools = [search]
agent = create_agent(model=llm, tools = tools)


result = agent.invoke({"messages":HumanMessage(content="What is the weather in Tokyo?")})

print(result)
    
