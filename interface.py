import gradio as gr
import requests
from typing import List, Tuple, Optional

# FastAPI endpoint URL
FASTAPI_URL = "http://localhost:8000/ask_doc_assistant"

def query_fastapi(message: str, history: Optional[List[Tuple[str, str]]], context: str) -> List[Tuple[str, str]]:
    if history is None:
        history = []

    try:
        # Prepare the payload
        payload = {
            "query": message,
            "user_context": context
        }
        
        # Send request to FastAPI
        response = requests.post(FASTAPI_URL, json=payload)
        response.raise_for_status()
        
        # Extract only the response text (not the entire object)
        response_data = response.json()

        # Ensure the response contains a valid message
        if "response" in response_data:
            bot_response = response_data["response"]
        elif "raw" in response_data:
            bot_response = response_data["raw"]
        else:
            bot_response = "⚠️ No valid response received from the server."

        # If the bot response is a dictionary, extract the text
        if isinstance(bot_response, dict) and "raw" in bot_response:
            bot_response = bot_response["raw"]

        # Ensure bot_response is a string
        bot_response = str(bot_response)

        history.append((message, bot_response))  # Append as a tuple (user message, bot response)
        return history
        
    except requests.exceptions.RequestException as e:
        error_msg = f"⚠️ Error: {str(e)}"
        history.append((message, error_msg))  # Append error message as a tuple
        return history

def create_gradio_interface():
    # Define theme and styling
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
    ).set(
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
                    height=400,
                    show_label=False,
                    bubble_full_width=False,
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your question here...",
                        scale=7,
                        show_label=False,
                        container=False,
                    )
                    submit_btn = gr.Button("Send", scale=1, variant="primary")

            with gr.Column(scale=1):
                context = gr.Textbox(
                    label="Additional Context",
                    placeholder="Add any relevant context...",
                    lines=3,
                    container=True,
                )
                clear_btn = gr.Button("Clear Chat", variant="secondary")

        # Event handlers
        msg.submit(query_fastapi, [msg, chatbot, context], chatbot).then(
            lambda: "", None, msg
        )
        submit_btn.click(query_fastapi, [msg, chatbot, context], chatbot).then(
            lambda: "", None, msg
        )

        clear_btn.click(
            lambda: [], None, chatbot, queue=False  # Properly clear the chat
        )

        # Example questions
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
    chat_interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
