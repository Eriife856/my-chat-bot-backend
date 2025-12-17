
from google import genai
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google.genai import types
import os
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
config = types.GenerateContentConfig(
    system_instruction='''You are a Child Development Assistant.
Your goal is to help with questions related to child development, child psychology,
early childhood education, parenting, and family studies.

Explain concepts clearly, in simple language, and use examples when possible.

If the user asks about something NOT related to child development, early childhood,
parenting, child psychology, or family studies, politely refuse and say:
"I can only help with child development related topics."
'''
)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)
class ChatRequest(BaseModel):
    message: str = Field(..., example="What is child development?")
@app.post("/chat")
def chat(input: ChatRequest):
    result = ""  # Ensure result is initialized
    try:
        response = client.models.generate_content_stream(
            model="gemini-2.0-flash", # Note: check if you meant 2.0; 2.5 is not out yet
            contents=input.message,
            config=config,
        )
        
        for chunks in response:
            if chunks.text:
                result += chunks.text
        
        return {"response": result}

    except Exception as e:
        # Instead of 'return e', return a structured error or a string
        return {"error": str(e)} 
        
        # OR better yet, use FastAPI's official way:
        # raise HTTPException(status_code=500, detail=str(e))