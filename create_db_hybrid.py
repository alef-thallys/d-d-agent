import json
import os
import shutil
import time

# --- IMPORTS CORRIGIDOS PARA O SEU AGENTE ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel

# --- CONFIGURAÇÃO ---
JSON_PATH = "rag_ready_manual.json"
CHROMA_PATH = "./dnd_db_2026"         # Mesmo diretório do agente
COLLECTION_NAME = "dnd_rules"         # Mesma coleção do agente
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

console = Console()

def main():
    console.clear()
    console.print(Panel.fit("[bold magenta]🔮 Criador de Grimório Vetorial (Local)[/bold magenta]", border_style="magenta"))

    # 1. Carregar o JSON
    if not os.path.exists(JSON_PATH):
        console.print(f"[bold red]❌ Erro: '{JSON_PATH}' não encontrado![/bold red]")
        console.print("[yellow]Execute o ingest_pdf.py primeiro.[/yellow]")
        return

    console.print(f"[cyan]📂 Carregando dados de: {JSON_PATH}...[/cyan]")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Converter para Documentos LangChain
    documents = []
    for item in data:
        doc = Document(
            page_content=item["content"],
            metadata=item["metadata"]
        )
        documents.append(doc)

    console.print(f"[green]✅ {len(documents)} fragmentos de regras carregados.[/green]")

    # 3. Limpar banco antigo para evitar duplicatas
    if os.path.exists(CHROMA_PATH):
        console.print(f"[yellow]🗑️ Removendo banco antigo em {CHROMA_PATH}...[/yellow]")
        shutil.rmtree(CHROMA_PATH)
        # Pequena pausa para o Windows liberar os arquivos
        time.sleep(1)

    # 4. Criar o Banco Vetorial (Isso roda na sua CPU/GPU, sem custo)
    console.print(f"[bold purple]🧠 Iniciando Embeddings ({EMBEDDING_MODEL})...[/bold purple]")
    console.print("[dim]Isso pode levar alguns minutos dependendo do seu processador...[/dim]")
    
    # Inicializa o modelo (baixa automaticamente na primeira vez)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Cria e salva o banco
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME  # <--- CRUCIAL: O agente busca esse nome
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