import os
from dotenv import load_dotenv
import openai

from app.constants import *

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv()

openai.api_key = os.getenv(OPENAI_API_KEY)


def chatgpt_response(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=250
    )
    response_dict = response.get('choices')

    if response_dict and len(response_dict) > 0:
        prompt_response = response_dict[0]["text"]
    return prompt_response


def chatgpt_turbo_response(message):
    template = """You are a chatbot to correct code.
    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    response = LLMChain(
        llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0),
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    prompt_response = response.predict(human_input=message)
    return prompt_response

