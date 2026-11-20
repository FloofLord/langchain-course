from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

from pydantic import BaseModel, Field


load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The url of the source")
    
    
class AgentResponse(BaseModel):
    """Schema for agent response with answers and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources: list[Source] = Field(default_factory=list, description="List of sources used to geerate the answer")



# Tavily has a langchain package that you can use instead of a custom tools
#from tavily import TavilyClient
# tavily = TavilyClient()
# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over the internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)


# gemma3:270m faster model but is not able to use tools
# llm = ChatOllama(temperature=0,model="gemma4:latest")

# gemini-4-argon/hydrogen
llm = ChatGoogleGenerativeAI(
    model="gemini-4-hydrogen",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


#tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools,response_format=AgentResponse)

#content = "What is the weather in Tokyo?"
content = "search for 3 job postings for a ai engineer using langchain remote on linkedin and list their details"

result = agent.invoke(
    {"messages": HumanMessage(content=content)}
)

print(result)
# bmad
# microsoft work iq
