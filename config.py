import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DB_NAME = os.getenv("DB_NAME", "customer_service_advanced.db")
    
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY not found in environment variables.")
