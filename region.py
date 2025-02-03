import os
import pinecone

# Load the Pinecone API key from the environment
api_key = os.getenv("PINECONE_API_KEY")

# Check if the API key is found
if not api_key:
    raise ValueError("PINECONE_API_KEY is not set. Please check your environment variables.")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=api_key)

# Check if it retrieves indexes
print(pc.list_indexes())  # Should show the list of indexes
