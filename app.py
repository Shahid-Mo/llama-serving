import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Simple FastAPI app
app = FastAPI()

# Config
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://llama-server:8080")

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7

@app.post("/chat")
async def chat(request: ChatRequest):
    # Format the prompt
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"<|system|>\n{msg.content}</s>\n"
        elif msg.role == "user":
            prompt += f"<|user|>\n{msg.content}</s>\n"
        elif msg.role == "assistant":
            prompt += f"<|assistant|>\n{msg.content}</s>\n"
    
    prompt += "<|assistant|>\n"
    
    # Call the LLM
    payload = {
        "prompt": prompt,
        "n_predict": request.max_tokens,
        "temperature": request.temperature,
        "stop": ["</s>", "<|user|>"]
    }
    
    try:
        response = requests.post(f"{LLAMA_SERVER_URL}/completion", json=payload)
        result = response.json()
        return {"content": result.get("content", "")}
    except Exception as e:
        return {"content": f"Error: {str(e)}"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)