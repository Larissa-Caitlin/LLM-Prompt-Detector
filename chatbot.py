import ollama

SYSTEM_PROMPT = """You are SecureBot — a helpful and intelligent assistant.
You can answer any question on any topic clearly and accurately.
You are security-aware, meaning you will not assist with illegal activities,
actual cyberattacks, or anything harmful.
Be concise, friendly, and helpful."""

def chat(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for turn in history:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})
    
    messages.append({"role": "user", "content": message})
    
    response = ollama.chat(
        model="phi3",
        messages=messages
    )
    
    return response["message"]["content"]