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

# Define the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible about outer space. If the question is not related to outer space, 
    just say, "This question is not about outer space".

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

# Asynchronous function to handle user input
async def user_input(user_question):
    chain = get_conversational_chain()
    response = chain.run({"question": user_question})
    return response

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat about Outer Space", layout="wide")
    st.header("Chat about Outer Space 🚀")

    user_question = st.text_input("Ask a Question about Outer Space")

    if user_question:
        response = asyncio.run(user_input(user_question))
        st.write("Gemini's Response: ", response)

if __name__ == "__main__":
    main()
