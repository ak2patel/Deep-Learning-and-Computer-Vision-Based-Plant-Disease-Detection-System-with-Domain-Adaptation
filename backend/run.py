import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run
import uvicorn

if __name__ == "__main__":
    print("[INFO] Starting AgriVision Backend...")
    print("[INFO] API will be available at: http://localhost:8000")
    print("[INFO] API Docs at: http://localhost:8000/docs")
    print()
    
    # Verify API key is loaded
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if api_key:
        print(f"[OK] Perplexity API key loaded: {api_key[:10]}...")
    else:
        print("[WARNING] Warning: No Perplexity API key found!")
    print()
    
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
