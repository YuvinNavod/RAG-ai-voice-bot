import os
from langchain_community.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import speech_recognition as sr
import pyttsx3
import random
import time

# Initialize the HuggingFace llm
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature": 0.7, "max_length": 500},  # Increased temperature for more diverse responses
    huggingfacehub_api_token="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"
)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

output_parser = StrOutputParser()

# Load the PDF document
loader = PyPDFLoader("laptop.pdf")
docs = loader.load()

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# Split the documents into chunks
splits = text_splitter.split_documents(docs)

# Create a vector store from the document chunks
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# Define prompt template
template = """
Answer this question using the provided context only.
{question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Set up the chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

# Initialize speech recognition and text-to-speech engines
r = sr.Recognizer()

# Use Google Cloud TTS or any advanced TTS service for more natural voice
# For simplicity, pyttsx3 is used here; replace it with Google TTS API, Amazon Polly, etc., for better voice quality
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Set to a more human-like voice
engine.setProperty('rate', 150)  # Adjust speaking rate for natural flow

def speech_to_text():
    """Converts speech to text using Google Speech Recognition"""
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio_text = r.listen(source)  # Listen to the user's speech
        try:
            recognized_text = r.recognize_google(audio_text)
            print(f"Recognized Text: {recognized_text}")
            return recognized_text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

def text_to_speech(text):
    """Converts text to speech using pyttsx3"""
    engine.say(text)
    engine.runAndWait()

# Small talk or alternative casual phrases
def casual_reply():
    responses = [
        "Hmm, let me think about that...",
        "Give me a moment...",
        "Interesting question, let me check.",
        "Let me look it up real quick.",
    ]
    text_to_speech(random.choice(responses))

# Initial greeting from the bot
def bot_greeting():
    greeting_text = "Hello! How can I assist you today?"
    print(f"Bot: {greeting_text}")
    text_to_speech(greeting_text)

# Farewell message from the bot
def bot_farewell():
    farewell_text = "You're welcome! Have a great day!"
    print(f"Bot: {farewell_text}")
    text_to_speech(farewell_text)

# Extract only the answer part
def extract_answer(full_response):
    """Extract the 'Answer' part from the response."""
    try:
        answer_start = full_response.index("Answer:") + len("Answer:")
        return full_response[answer_start:].strip()  # Only return the answer part
    except ValueError:
        return full_response  # If "Answer:" is not found, return the full response

# Main loop for interaction
def start_chat():
    bot_greeting()  # Bot starts with a greeting

    while True:
        # Get the user's question (speech-to-text)
        user_question = speech_to_text()
        
        if user_question:
            # If the user says "thank you" or similar, end the conversation
            if "thank you" in user_question.lower():
                bot_farewell()  # End the chat
                break

            # Insert pause to simulate thinking
            casual_reply()  # Simulate human-like response

            time.sleep(random.uniform(1, 2))  # Add natural delay for a thinking effect

            # Example usage of the chain: pass the user's question to the chain
            full_response = chain.invoke(user_question)

            # Extract only the answer part
            answer = extract_answer(full_response)

            # Convert the answer back to speech (text-to-speech)
            print(f"Bot: {answer}")
            text_to_speech(answer)
        else:
            # If no valid input is detected, ask again
            text_to_speech("Sorry, I did not get that. Could you please repeat?")
            continue

# Start the chat
start_chat()
