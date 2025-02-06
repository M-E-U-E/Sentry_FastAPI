import gradio as gr
import requests
from typing import List, Tuple, Optional
import json
from pinecone import Pinecone, PodSpec
from sentence_transformers import SentenceTransformer
import os
import uuid
from dotenv import load_dotenv
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Sentence Transformer for 768-dimensional embeddings
encoder = SentenceTransformer('paraphrase-mpnet-base-v2')

# FastAPI endpoint URL
FASTAPI_URL = "http://localhost:8000/ask_doc_assistant"

# Admin password hash (store the hash of your password in .env)
ADMIN_PASSWORD_HASH = os.getenv('ADMIN_PASSWORD_HASH')

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)

# Use the 'qa' index for storing the QA pairs
index = pc.Index("qa")

def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(input_password: str) -> bool:
    """Verify if input password matches stored hash."""
    return hash_password(input_password) == ADMIN_PASSWORD_HASH

def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using Sentence Transformer."""
    try:
        return encoder.encode(text).tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

def find_similar_question(question: str, threshold: float = 0.85) -> Optional[Tuple[str, str]]:
    """Find similar question and return answer and vector ID if found."""
    try:
        question_embedding = get_embedding(question)
        
        results = index.query(
            vector=question_embedding,
            top_k=1,
            include_metadata=True,
            filter={"type": "qa_pair"}
        )

        if results['matches'] and results['matches'][0]['score'] > threshold:
            return (
                results['matches'][0]['metadata'].get('answer'),
                results['matches'][0]['id']
            )
        return None
    except Exception as e:
        logger.error(f"Error finding similar question: {e}")
        return None

def store_qa_pair(question: str, answer: str) -> Optional[str]:
    """Store question-answer pair in Pinecone and return vector ID."""
    try:
        if answer and "⚠️ Error" not in answer:
            question_embedding = get_embedding(question)
            vector_id = str(uuid.uuid4())
            
            index.upsert(vectors=[{
                'id': vector_id,
                'values': question_embedding,
                'metadata': {
                    'type': 'qa_pair',
                    'question': question,
                    'answer': answer,
                    'doc_name': f'doc_{vector_id}'
                }
            }])
            return vector_id
        return None
    except Exception as e:
        logger.error(f"Error storing QA pair: {e}")
        return None

def delete_all_qa_pairs(password: str) -> Tuple[bool, str]:
    """Delete all QA pairs from the 'qa' index if password is correct."""
    try:
        if not verify_password(password):
            return False, "❌ Invalid admin password"
        
        # Query to get all QA pair IDs
        results = index.query(
            vector=[0] * 768,  # Dummy vector for metadata-only query
            top_k=10000,  # Adjust based on your maximum expected QA pairs
            filter={"type": "qa_pair"},
            include_metadata=False
        )
        
        if not results['matches']:
            return True, "ℹ️ No QA pairs found to delete"
        
        # Extract all vector IDs
        vector_ids = [match['id'] for match in results['matches']]
        logger.info(f"Found {len(vector_ids)} QA pairs to delete.")

        # Try deleting the QA pairs by their IDs
        try:
            delete_response = index.delete(ids=vector_ids)  # Delete by IDs
            logger.info(f"Deletion response: {delete_response}")

            # Verify deletion by querying again
            verification = index.query(
                vector=[0] * 768,
                top_k=10000,
                filter={"type": "qa_pair"},
                include_metadata=False
            )
            
            if verification['matches']:
                return False, f"❌ Deletion failed - {len(verification['matches'])} QA pairs still exist"
            
            return True, f"✅ Successfully deleted {len(vector_ids)} QA pairs"
            
        except Exception as delete_error:
            logger.error(f"Error during deletion operation: {delete_error}")
            return False, f"❌ Error during deletion: {str(delete_error)}"
        
    except Exception as e:
        logger.error(f"Error deleting all QA pairs: {e}")
        return False, f"❌ Error deleting QA pairs: {str(e)}"


def query_fastapi(message: str, history: List[dict], context: str) -> List[dict]:
    """Query FastAPI endpoint and manage local cache."""
    if history is None:
        history = []

    try:
        # Check cache first
        cached_result = find_similar_question(message)
        if cached_result:
            answer, vector_id = cached_result
            history.append({"role": "user", "content": message})
            history.append({
                "role": "assistant",
                "content": f"[Cached Response] {answer}\n\nVector ID: {vector_id}"
            })
            return history

        # Query FastAPI if no cache hit
        payload = {"query": message, "user_context": context}
        response = requests.post(FASTAPI_URL, json=payload)
        response.raise_for_status()
        response_data = response.json()

        bot_response = response_data.get("response", "⚠️ No valid response received from the server.")

        if isinstance(bot_response, dict):
            bot_response = bot_response.get("raw", "⚠️ Missing data in the response.")

        # Store new QA pair if valid
        vector_id = None
        if "⚠️ Error" not in bot_response and "⚠️ Missing data" not in bot_response:
            vector_id = store_qa_pair(message, bot_response)
        
        bot_response = str(bot_response)
        if vector_id:
            bot_response += f"\n\nVector ID: {vector_id}"

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_response})
        return history

    except requests.exceptions.RequestException as e:
        error_msg = f"⚠️ Error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history

def create_gradio_interface():
    """Create Gradio interface with admin features."""
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
                    type="messages"
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
                
                # Admin section for QA management
                gr.Markdown("### Admin Controls")
                with gr.Tab("Delete Single QA"):
                    vector_id = gr.Textbox(
                        label="Vector ID",
                        placeholder="Enter Vector ID to delete..."
                    )
                    admin_password_single = gr.Textbox(
                        label="Admin Password",
                        placeholder="Enter admin password...",
                        type="password"
                    )
                    delete_btn = gr.Button("Delete QA Pair", variant="secondary")
                
                with gr.Tab("Delete All QA"):
                    admin_password_bulk = gr.Textbox(
                        label="Admin Password",
                        placeholder="Enter admin password...",
                        type="password"
                    )
                    delete_all_btn = gr.Button("Delete All QA Pairs", variant="secondary")
                    gr.Markdown("⚠️ Warning: This will delete all QA pairs permanently!")

                # Status message for admin actions
                status_msg = gr.Markdown("")

        def handle_delete_single(vector_id: str, password: str) -> str:
            """Handle deletion of single QA pair."""
            if not vector_id:
                return "⚠️ Please provide a Vector ID"
            
            if not password:
                return "⚠️ Please provide the admin password"
            
            if not verify_password(password):
                return "❌ Invalid admin password"
            
            try:
                index.delete(ids=[vector_id])
                return "✅ QA pair deleted successfully"
            except Exception as e:
                logger.error(f"Error deleting QA pair: {e}")
                return f"❌ Failed to delete QA pair: {str(e)}"

        def handle_delete_all(password: str) -> str:
            """Handle deletion of all QA pairs."""
            logger.info("Attempting to delete all QA pairs.")
            success, message = delete_all_qa_pairs(password)
            
            if success:
                logger.info("All QA pairs deleted successfully.")
            else:
                logger.error(f"Error deleting all QA pairs: {message}")
            
            return message


        # Event handlers
        msg.submit(query_fastapi, [msg, chatbot, context], chatbot).then(
            lambda: "", None, msg
        )
        submit_btn.click(query_fastapi, [msg, chatbot, context], chatbot).then(
            lambda: "", None, msg
        )
        clear_chat_btn.click(lambda: [], None, chatbot, queue=False)
        delete_btn.click(
            handle_delete_single,
            [vector_id, admin_password_single],
            status_msg
        )
        delete_all_btn.click(
            handle_delete_all,
            [admin_password_bulk],
            status_msg
        )

        gr.Examples(
            examples=[
                ["What is the role of the CRON job in this project?"],
                ["What are the bacground and project goal of this Django project?"],
                ["How can I customize the Django admin interface for specific use cases?"],
                ["How to configure environment variables and database of this project?"],
                ["How are PostgreSQL and Django integrated in this project?"],
            ],
            inputs=msg,
            label="Try these common questions:"
        )

    return chat_interface

if __name__ == "__main__":
    chat_interface = create_gradio_interface()
    chat_interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True
    )
