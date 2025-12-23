# llm.py
import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Model and system prompt
MODEL = "gpt-4o-mini"  # change if needed
SYSTEM_PROMPT = (
    "You are a concise financial assistant. "
    "Use the context provided to answer financial questions. "
    "If the question is general knowledge or not finance-related, answer it accurately based on general knowledge."
)

def chat_with_context(user_prompt, system_prompt=SYSTEM_PROMPT, max_tokens=500):
    """
    Sends a user prompt to the OpenAI chat model with context and returns the response.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Use new OpenAI SDK interface (>=1.0.0)
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0
    )
    
    # Extract the answer
    answer = resp.choices[0].message.content
    return answer

# Quick test
if __name__ == "__main__":
    test_prompt = "CONTEXT: Hello world\nQUESTION: Summarize context"
    print(chat_with_context(test_prompt))
