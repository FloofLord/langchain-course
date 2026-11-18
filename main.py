from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

def main():
    information = """The Nova Scotia Duck Tolling Retriever is a medium-sized gun dog bred primarily for hunting. It is often referred to as a "toller". It is the smallest of the retrievers, and is often mistaken for a small Golden Retriever. Tollers are intelligent, eager to please, alert, and energetic. The toller also has webbed feet.[1] The name "toller" is derived from their ability to lure waterfowl within gunshot range. The dog goes up to the water's edge and attracts the ducks so the hunter can shoot them. Then they go to retrieve the duck from the water. The breed originated in Yarmouth County, Nova Scotia, Canada.[2] The American Kennel Club ranks the toller as the 69th most popular dog breed.[1]"""
    
    # Cleaned up template formatting so instructions and data are distinctly separated
    summary_template = """Based on the information below about a dog, create:
1. A short summary
2. Two interesting facts about them

Information:
{information}"""
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )
    
    # Kept your exact setup with no temperature parameter added
    # For google models you dont have to set temprature
    # Previously I was usng google hydrogen which is agentic and you can't pass a string to a agentic model like that
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-3.8-flash",
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    # )
    model = "gpt-oss:20b" #"gemma3:270m"
    llm = ChatOllama(temperature=0, model=model)

    chain = summary_prompt_template | llm  
    response = chain.invoke(input={"information": information})  
    print(response.content)
    

if __name__ == "__main__":
    main()
