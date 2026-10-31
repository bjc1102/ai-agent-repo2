import os
from  pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

env_path = Path(__file__).resolve().parent

load_dotenv(env_path / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key or not api_key.strip():
    raise ValueError("OPENAI_API_KEY 설정이 필요합니다.")

client = OpenAI(api_key=api_key)




