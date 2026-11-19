from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

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
    return tavily.search(query=query)

# gemma3:270m faster model but is not able to use tools
#llm = ChatOllama(temperature=0,model="gemma4:latest")

#gemini-4-argon/hydrogen
llm = ChatGoogleGenerativeAI(
    model="gemini-4-hydrogen",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


tools = [search]
agent = create_agent(model=llm, tools = tools)


result = agent.invoke({"messages":HumanMessage(content="What is the weather in Tokyo?")})

print(result)
    
