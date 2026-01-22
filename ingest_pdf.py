import os

# --- IMPORTS MODERNOS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 

# --- INTERFACE VISUAL ---
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme
from tqdm import tqdm

# --- CONFIGURAÇÃO ---
# Nome exato do arquivo PDF (deve estar na mesma pasta)
PDF_PATH = "dd-5e-livro-do-jogador.pdf"
DB_DIR = "./dnd_db_2026"

# Configuração visual
custom_theme = Theme({"info": "cyan", "warning": "yellow", "error": "bold red", "success": "bold green"})
console = Console(theme=custom_theme)

def processar_pdf():
    console.print(Panel.fit(f"📘 Iniciando Ingestão de Grimório: [bold]{PDF_PATH}[/bold]", style="blue"))
    
    if not os.path.exists(PDF_PATH):
        console.print(f"[error]❌ Erro: O arquivo '{PDF_PATH}' não foi encontrado na pasta raiz.[/error]")
        console.print("[warning]→ Verifique o nome do arquivo ou se ele está na pasta correta.[/warning]")
        return []

    # 1. Carrega o PDF
    with console.status("[bold blue]Lendo páginas do PDF...[/bold blue]", spinner="dots"):
        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()
    
    console.print(f"[success]✅ PDF carregado: {len(pages)} páginas encontradas.[/success]")

    # 2. Configura o "Cortador" (Splitter)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    # 3. Corta em chunks
    with console.status("[bold purple]Fatiando o conhecimento em pedaços mágicos...[/bold purple]", spinner="moon"):
        chunks = text_splitter.split_documents(pages)
        
        # Adiciona metadados
        for chunk in chunks:
            chunk.metadata["source"] = "Livro do Jogador (PDF)"
            chunk.metadata["type"] = "pdf_content"

    console.print(f"[info]📦 Total de chunks gerados:[/info] [bold white]{len(chunks)}[/bold white]")
    return chunks

def main():
    # 1. Processa o Arquivo
    novos_documentos = processar_pdf()

    if not novos_documentos:
        return

    # 2. Carrega Embeddings
    with console.status("[bold green]Carregando modelo de Embeddings...[/bold green]", spinner="dots"):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    console.print(f"[info]💾 Conectando ao banco de dados em '{DB_DIR}'...[/info]")
    
    # 3. Inicializa o Banco (Modo Append)
    vector_db = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=embedding_model,
        collection_name="dnd_rules" # Importante: Usar a mesma collection do create_db
    )
    
    # 4. Adiciona em Lotes com Barra de Progresso
    batch_size = 100
    total_docs = len(novos_documentos)
    
    console.print("[bold cyan]Iniciando assimilação de conhecimento...[/bold cyan]")
    
    with tqdm(total=total_docs, desc="Indexando", unit="chunk", colour="green") as pbar:
        for i in range(0, total_docs, batch_size):
            batch = novos_documentos[i:i + batch_size]
            vector_db.add_documents(batch)
            pbar.update(len(batch))
            
    console.print(Panel(f"✅ SUCESSO!\nO conteúdo do PDF foi adicionado ao grimório em: {DB_DIR}", style="bold green"))

if __name__ == "__main__":
    main()