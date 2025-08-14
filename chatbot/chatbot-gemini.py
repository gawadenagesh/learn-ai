# Streamlite framework
import streamlit as streamlit
import os  # For environment variables
from langchain_google_genai import ChatGoogleGenerativeAI  # Updated import
from langchain.schema import HumanMessage

os.environ["GEMINI_API_KEY"] = "AIzaSyDtg-S9E3U-KlwrfJJ9KlG-dvKt9KI1I6E"

streamlit.title("ChatBot: With Langchain & Gemini")

# Get API key from environment variable
gemini_api_key = os.environ.get("GEMINI_API_KEY")

if not gemini_api_key:
    streamlit.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
else:
    # Model selection (optional)
    model_name = 'gemini-1.5-flash' 
    #streamlit.selectbox("Choose a Gemini model:", ["gemini-1.5-flash", "gemini-1.5-pro"])

    input_text = streamlit.text_input("Search for a topic:")

    if input_text:
        streamlit.write(f"You searched for: {input_text}")

        try:
            chat = ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_api_key)  # Pass API key
            response = chat([HumanMessage(content=input_text)])
            streamlit.write(f"Response from Gemini: {response.content}")
        except Exception as e:
            streamlit.error(f"An error occurred: {e}")

# How to use memory with Langchain and with above code
# from langchain.memory import ConversationBufferMemory
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# chat = ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_api_key, memory=memory)  # Pass memory
# response = chat([HumanMessage(content=input_text)])
# streamlit.write(f"Response from Gemini with memory: {response.content}")

# Note: Make sure to replace the API key with your actual Gemini API key.
# How to use pinecone vector database with Langchain and with above code
# from langchain.vectorstores import Pinecone
# from langchain.embeddings import OpenAIEmbeddings
# import pinecone

# how to use Pinecone with Langchain


# how to use RAG with Langchain and gemini
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import Pinecone
# from langchain.embeddings import OpenAIEmbeddings