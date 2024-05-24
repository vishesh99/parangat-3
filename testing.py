from flask import Flask, request, jsonify
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()

# Retrieve environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if environment variables are loaded correctly
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set correctly")

# Flask app
app = Flask(__name__)


def get_conversational_chain():
    """
    Create and return the conversational chain for the language model.
    """
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


async def generate_response(user_question):
    """
    Generate a response for the given user question using the conversational chain.
    """
    chain = get_conversational_chain()
    response = chain.run({"question": user_question})
    return response


def process_user_question(data):
    """
    Extract the user question from the request data.
    """
    return data.get('question')


def create_response_message(response):
    """
    Create a JSON response message from the generated response.
    """
    return jsonify({"response": response})


@app.route('/api/chat', methods=['POST'])
def chat_api():
    """
    API endpoint to handle chat requests.
    """
    if request.method == 'POST':
        data = request.json
        user_question = process_user_question(data)

        if user_question:
            response = asyncio.run(generate_response(user_question))
            return create_response_message(response)
        else:
            return jsonify({"error": "No question provided"}), 400
    else:
        return jsonify({"error": "Only POST requests are supported"}), 405


if __name__ == "__main__":
    app.run()
