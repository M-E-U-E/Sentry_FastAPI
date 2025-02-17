import os
from dotenv import load_dotenv
import google.generativeai as genai

def test_gemini_api():
    # Load environment variables
    load_dotenv()
    
    # Get the API key from environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file")
        return False
    
    try:
        # Configure the Gemini API with your key
        genai.configure(api_key=api_key)
        
        # Create a simple model instance
        model = genai.GenerativeModel('gemini-pro')
        
        # Try a simple generation request
        response = model.generate_content("Hello, can you tell me if my API key is working?")
        
        # Check if we got a valid response
        if response and hasattr(response, 'text'):
            print(f"✅ SUCCESS: API key is working! Response received:")
            print(f"\n{response.text}\n")
            return True
        else:
            print("❌ ERROR: Received an empty or invalid response")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to Gemini API: {str(e)}")
        return False

if __name__ == "__main__":
    test_gemini_api()