import os
import json
import logging

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import MultiQueryRetriever 
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live

logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

load_dotenv()
console = Console()

DB_DIR = "./dnd_db_2026"
JSON_PATH = "rag_v5.json" 
GEMINI_MODEL = "gemini-flash-latest"

def setup_agent_pro():
    with console.status("[bold purple]🔮 Conjurando arquitetura Híbrida...[/bold purple]", spinner="moon"):
        
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.2)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings, collection_name="dnd_rules")
        chroma_retriever = vector_db.as_retriever(search_kwargs={"k": 4})

        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in data]
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 4
        else:
            console.print("[red]⚠️ JSON não encontrado. Rodando apenas com Vetorial.[/red]")
            bm25_retriever = None

        if bm25_retriever:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[chroma_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
        else:
            ensemble_retriever = chroma_retriever

        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=ensemble_retriever,
            llm=llm
        )

        # Prompt do Mestre
        system_prompt = (
            "Você é um Mestre de D&D 5ª Edição sábio e preciso.\n"
            "Use APENAS o contexto fornecido para responder.\n"
            "Se o contexto tiver o título da seção (ex: 'Capítulo 3: Classes > Guerreiro'), use isso para saber de quem é a regra.\n"
            "Se a resposta não estiver no contexto, diga que não sabe. Não alucine regras.\n"
            "Responda em Português do Brasil com formatação Markdown clara."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "CONTEXTO RECUPERADO:\n{context}\n\nPERGUNTA:\n{question}")
        ])

        def format_docs(docs):
            return "\n\n".join([f"[{doc.metadata.get('chapter', 'Geral')}]: {doc.page_content}" for doc in docs])

        chain = (
            RunnablePassthrough.assign(context=(lambda x: x["question"]) | multi_query_retriever | format_docs)
            | prompt
            | llm
            | StrOutputParser()
        )

        chat_history = ChatMessageHistory()
        
        return RunnableWithMessageHistory(
            chain, # pyright: ignore[reportArgumentType]
            lambda session_id: chat_history,
            input_messages_key="question",
            history_messages_key="history"
        )

def main():
    console.clear()
    console.print(Panel.fit("[bold yellow]🐉 D&D RAG PRO (Híbrido + Multi-Query)[/bold yellow]", border_style="red"))
    
    agent = setup_agent_pro()
    
    console.print("[dim]Sistema pronto. Digite 'sair' para encerrar.[/dim]\n")

    while True:
        user_input = console.input("[bold cyan]🧙 Pergunta:[/bold cyan] ")
        if user_input.lower() in ["sair", "exit"]: break
        
        with Live(Panel("Consultando os planos...", title="Mestre", border_style="green"), refresh_per_second=10) as live:
            response = agent.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": "mesa_pro"}}
            )
            live.update(Panel(Markdown(response), title="Mestre", border_style="green"))

if __name__ == "__main__":
    main()