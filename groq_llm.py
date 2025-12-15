import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_answer(context, question):
    prompt = f"""
    নিচের তথ্য ব্যবহার করে প্রশ্নের উত্তর দাও।
    যদি উত্তর না পাও, বলো:
    "দুঃখিত, এই প্রশ্নের উত্তর আমার জানা নেই।"

    তথ্য:
    {context}

    প্রশ্ন:
    {question}

    বাংলায় উত্তর দাও।
    """

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content
