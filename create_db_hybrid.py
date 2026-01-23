import json
import os
import shutil
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel

JSON_PATH = "rag_v5.json"
CHROMA_PATH = "./dnd_db_2026"        
COLLECTION_NAME = "dnd_rules"         
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

console = Console()

def main():
    console.clear()
    console.print(Panel.fit("[bold magenta]🔮 Criador de Grimório Vetorial (Local)[/bold magenta]", border_style="magenta"))

    if not os.path.exists(JSON_PATH):
        console.print(f"[bold red]❌ Erro: '{JSON_PATH}' não encontrado![/bold red]")
        console.print("[yellow]Execute o ingest_pdf.py primeiro.[/yellow]")
        return

    console.print(f"[cyan]📂 Carregando dados de: {JSON_PATH}...[/cyan]")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        doc = Document(
            page_content=item["content"],
            metadata=item["metadata"]
        )
        documents.append(doc)

    console.print(f"[green]✅ {len(documents)} fragmentos de regras carregados.[/green]")

    if os.path.exists(CHROMA_PATH):
        console.print(f"[yellow]🗑️ Removendo banco antigo em {CHROMA_PATH}...[/yellow]")
        shutil.rmtree(CHROMA_PATH)
        time.sleep(1)

    console.print(f"[bold purple]🧠 Iniciando Embeddings ({EMBEDDING_MODEL})...[/bold purple]")
    console.print("[dim]Isso pode levar alguns minutos dependendo do seu processador...[/dim]")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME  
    )

    console.print(Panel(
        f"[bold green]🚀 SUCESSO![/bold green]\n\n"
        f"Banco salvo em: [white]{CHROMA_PATH}[/white]\n"
        f"Coleção: [white]{COLLECTION_NAME}[/white]\n"
        f"Documentos: [white]{len(documents)}[/white]",
        title="Grimório Criado",
        border_style="green"
    ))
    console.print("[bold cyan]Agora você pode rodar o seu agente![/bold cyan]")

if __name__ == "__main__":
    main()