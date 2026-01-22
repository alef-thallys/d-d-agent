import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv 

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    print("ERRO: A variável GOOGLE_API_KEY não foi encontrada no arquivo .env")
else:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("🔮 Invocando o Mestre dos Magos...")

# 1. Carregar o Banco Vetorial Existente
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_db = Chroma(persist_directory="./dnd_db_hybrid_2024", embedding_function=embedding_model)

# 2. Configurar o Modelo (Gemini 1.5 Flash - Rápido e Grátis)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

def buscar_regras(pergunta):
    # Recupera os 100 documentos mais relevantes
    # O modelo multilíngue entende que "Bola de Fogo" = "Fireball"
    docs = vector_db.similarity_search(pergunta, k=100)
    return docs

def gerar_resposta(pergunta, contexto_docs):
    contexto_texto = "\n\n---\n\n".join([d.page_content for d in contexto_docs])
    
    prompt = f"""
    Você é um Mestre de D&D 5ª Edição experiente e prestativo.
    
    INSTRUÇÕES:
    1. O usuário fará perguntas em PORTUGUÊS.
    2. Use o CONTEXTO abaixo (que está em Inglês) como fonte da verdade absoluta.
    3. Responda em PORTUGUÊS.
    4. Ao citar termos técnicos (Magias, Habilidades), use o termo em Português e coloque o original em inglês entre parênteses na primeira vez. 
       Ex: "Você usa Mãos Mágicas (Mage Hand)..."
    5. Se a resposta exigir cálculo (dano, acerto), explique a fórmula.

    CONTEXTO (Regras em Inglês):
    {contexto_texto}

    PERGUNTA DO JOGADOR: 
    {pergunta}
    
    RESPOSTA DO MESTRE:
    """
    
    resposta = llm.invoke(prompt)
    return resposta.content

#Loop do Chat
print("\n--- RAG D&D 5e (Base: GitHub English | Chat: Português) ---")
print("Digite 'sair' para encerrar.\n")

while True:
    user_input = input("🧙 Pergunta: ")
    if user_input.lower() in ["sair", "exit"]: break
    
    # 1. Retrieval
    print("   (Consultando grimório...)", end="\r")
    docs = buscar_regras(user_input)
    
    # 2. Generation
    resposta = gerar_resposta(user_input, docs)
    
    print(f"\n📜 Mestre: {resposta}\n")
    print("-" * 50)