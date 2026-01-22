import os
import google.generativeai as genai
from dotenv import load_dotenv 

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    print("ERRO: A variável GOOGLE_API_KEY não foi encontrada no arquivo .env")
else:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("Procurando modelos disponíveis...")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(f"- {m.name}")