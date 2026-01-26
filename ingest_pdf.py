import os
import re
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.theme import Theme

LIB_DIR = "./biblioteca"
OUTPUT_JSON = "rag_v5.json" 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SECTION_PATTERN = re.compile(r"(?:^|\n)\s*([A-ZÃÁÂÊÉÍÕÓÚÇ][A-ZÃÁÂÊÉÍÕÓÚÇ\s\-:]{3,})(?:\n|$)")

console = Console(theme=Theme({"info": "cyan", "success": "bold green", "warning": "yellow", "error": "bold red"}))

def clean_filename(filename):
    name = os.path.splitext(filename)[0]
    name = name.replace("-", " ").replace("_", " ").title()
    return name

def clean_text(text):
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    return text

def process_pdf(file_path):
    filename_display = clean_filename(os.path.basename(file_path))
    console.print(Panel(f"📘 Processando: {filename_display}", style="blue"))
    
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    except Exception as e:
        console.print(f"[error]Erro ao ler {file_path}: {e}[/error]")
        return []

    full_text = "\n".join([p.page_content for p in pages])
    full_text = clean_text(full_text)

    fragments = SECTION_PATTERN.split(full_text)
    rag_docs = []
    current_section = "Introdução/Geral"
    
    if len(fragments) == 1:
        content_parts = [fragments[0]]
        section_titles = [current_section]
    else:
        content_parts = [fragments[0]]
        section_titles = [current_section]
        
        for i in range(1, len(fragments), 2):
            title = fragments[i].strip()
            content = fragments[i+1] if i+1 < len(fragments) else ""
            
            section_titles.append(title.title()) 
            content_parts.append(content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    total_chunks_file = 0
    
    for section_title, content in zip(section_titles, content_parts):
        if not content.strip(): continue

        chunks = splitter.split_text(content)
        
        for chunk in chunks:
            if len(chunk) < 50: continue 
            
            contextualized_content = f"Livro: {filename_display}\nSeção: {section_title}\n---\n{chunk}"
            
            rag_docs.append({
                "content": contextualized_content,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "book_name": filename_display,
                    "section": section_title,
                    "original_content": chunk
                }
            })
            total_chunks_file += 1

    console.print(f"[info]   -> {len(pages)} páginas lidas.[/info]")
    console.print(f"[success]   -> {total_chunks_file} chunks gerados.[/success]")
    
    return rag_docs

def main():
    if not os.path.exists(LIB_DIR):
        console.print(f"[error]Pasta {LIB_DIR} não encontrada! Crie-a e coloque seus PDFs lá.[/error]")
        return

    pdf_files = [f for f in os.listdir(LIB_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        console.print(f"[warning]Nenhum PDF encontrado em {LIB_DIR}.[/warning]")
        return

    all_docs = []
    console.print(f"[bold]📚 Iniciando ingestão de {len(pdf_files)} livros...[/bold]")

    for pdf in pdf_files:
        docs = process_pdf(os.path.join(LIB_DIR, pdf))
        all_docs.extend(docs)

    console.print(Panel(f"💾 Salvando banco de dados JSON...", style="yellow"))
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    console.print(Panel(
        f"[bold green]CONCLUÍDO![/bold green]\n"
        f"Total de Fragmentos: {len(all_docs)}\n"
        f"Arquivo gerado: {OUTPUT_JSON}\n\n"
        f"[white]Próximo passo: Rode 'python create_db_hybrid.py'[/white]",
        title="Ingestão Finalizada",
        border_style="green"
    ))

if __name__ == "__main__":
    main()