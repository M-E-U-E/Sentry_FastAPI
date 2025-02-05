import gradio as gr
import requests
from typing import List, Tuple, Optional
import json
from pinecone import Pinecone, PodSpec
from sentence_transformers import SentenceTransformer
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Sentence Transformer for 768-dimensional embeddings
encoder = SentenceTransformer('paraphrase-mpnet-base-v2')  # Outputs 768-dimensional vectors

# FastAPI endpoint URL
FASTAPI_URL = "http://localhost:8000/ask_doc_assistant"

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)
index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))

# Generate 768-dimensional embedding
def get_embedding(text: str) -> List[float]:
    return encoder.encode(text).tolist()

# Find similar question based on embedding similarity
def find_similar_question(question: str, threshold: float = 0.85) -> Optional[str]:
    question_embedding = get_embedding(question)
    
    results = index.query(
        vector=question_embedding,
        top_k=1,
        include_metadata=True,
        filter={"type": "qa_pair"}
    )

    if results['matches'] and results['matches'][0]['score'] > threshold:
        return results['matches'][0]['metadata'].get('answer')
    return None

# Store question-answer pair in Pinecone
def store_qa_pair(question: str, answer: str):
    if answer and "⚠️ Error" not in answer:
        question_embedding = get_embedding(question)
        vector_id = str(uuid.uuid4())
        
        index.upsert(vectors=[
            {
                'id': vector_id,
                'values': question_embedding,
                'metadata': {
                    'type': 'qa_pair',
                    'question': question,
                    'answer': answer,
                    'doc_name': f'doc_{vector_id}'  # Ensure doc_name is always present
                }
            }
        ])

# Query FastAPI and manage local cache
def query_fastapi(message: str, history: Optional[List[Tuple[str, str]]], context: str) -> List[Tuple[str, str]]:
    if history is None:
        history = []

    try:
        cached_answer = find_similar_question(message)
        if cached_answer:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"[Cached Response] {cached_answer}"})
            return history

        payload = {"query": message, "user_context": context}
        response = requests.post(FASTAPI_URL, json=payload)
        response.raise_for_status()
        response_data = response.json()

        bot_response = response_data.get("response", "⚠️ No valid response received from the server.")

        if isinstance(bot_response, dict):
            bot_response = bot_response.get("raw", "⚠️ Missing data in the response.")

        if "⚠️ Error" not in bot_response and "⚠️ Missing data" not in bot_response:
            store_qa_pair(message, bot_response)
        
        bot_response = str(bot_response)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_response})
        return history

    except requests.exceptions.RequestException as e:
        error_msg = f"⚠️ Error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history

# Gradio Interface
def create_gradio_interface():
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray").set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        button_primary_text_color="white",
    )

    with gr.Blocks(theme=theme, css="footer {display: none !important}") as chat_interface:
        gr.Markdown("""
        # 📚 Documentation Assistant
        Ask questions about the documentation and get helpful responses. Provide additional context if needed.
        """)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=600,
                    show_label=False,
                    bubble_full_width=False,
                    container=True,
                    scale=2,
                    type="messages"  # Updated to avoid deprecation warning
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your question here...",
                        scale=7,
                        show_label=False,
                        container=True,
                        lines=2
                    )
                    submit_btn = gr.Button("Send", scale=1, variant="primary")

            with gr.Column(scale=1):
                context = gr.Textbox(
                    label="Additional Context",
                    placeholder="Add any relevant context...",
                    lines=3,
                    value="I'm a new developer trying to understand the system."
                )
                clear_chat_btn = gr.Button("Clear Chat", variant="secondary")

        # Event handlers
        msg.submit(query_fastapi, [msg, chatbot, context], chatbot).then(lambda: "", None, msg)
        submit_btn.click(query_fastapi, [msg, chatbot, context], chatbot).then(lambda: "", None, msg)
        clear_chat_btn.click(lambda: [], None, chatbot, queue=False)

        gr.Examples(
            examples=[
                ["How to run this Django project?"],
                ["What are the main features and project goal?"],
                ["How to setup the database of this project?"],
                ["How to configure environment variables?"],
                ["What dependencies are required for production?"],
            ],
            inputs=msg,
            label="Try these common questions:"
        )

    return chat_interface

if __name__ == "__main__":
    chat_interface = create_gradio_interface()
    chat_interface.launch(share=True, server_name="0.0.0.0", server_port=7861, show_error=True)
