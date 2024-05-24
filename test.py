import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables
load_dotenv()

# Retrieve environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if environment variables are loaded correctly
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set correctly")

def get_chain():
    prompt_template="""
    Answer the question as detailed as possible about outer space. If the question is not related to outer space, 
    just say, "This question is not about outer space".

    Question:
    {question}

    """

    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.6)
    prompt=PromptTemplate(template=prompt_template,input_variables=['question'])
    chain=LLMChain(llm=model,prompt=prompt)

    return chain

def ans(question):
    chain=get_chain()
    response=chain.run({"question":question})
    return response

def main():
    st.set_page_config(page_title="chat about OuterSpace" ,layout="wide")
    st.header("chat about OuterSpace")

    user_question=st.text_input("ask the question")

    response=ans(user_question)

    st.write("answer:",response)

if __name__=="__main__":
    main()