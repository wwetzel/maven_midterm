# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

out_fp = './data/'
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local(out_fp + 'nvidia_10k_faiss_index.bin', embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()
openai_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# ChatOpenAI Templates
template = """Answer the question based only on the following context. If you cannot answer the question with the context, respond with 'I don't know'. You'll get a big bonus and a potential promotion if you provide a high quality answer:

Context:
{context}

Question:
{question}
"""
prompt_template = ChatPromptTemplate.from_template(template)
retrieval_augmented_qa_chain_openai = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt_template | openai_llm, "context": itemgetter("context")}
)


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 250,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    question = message.content

    response = retrieval_augmented_qa_chain_openai.invoke({"question": question})
    response_content = response["response"].content
    combined_context = '\n'.join([document.page_content for document in response["context"]])
    page_numbers = set([document.metadata['page'] for document in response["context"]])

    msg = cl.Message(content=response_content)
    await msg.send()
