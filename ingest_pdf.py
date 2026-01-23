import os
import re
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

# --- CONFIGURAÇÃO ---
LIB_DIR = "./biblioteca"
OUTPUT_JSON = "rag_ready_v2.json" # Novo arquivo para não sobrescrever o antigo
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PAGE_OFFSET = -1 # Ajuste conforme seu PDF

CHAPTER_MAP = {
    "Capítulo 1: Criação de Personagem": 11,   
    "Capítulo 2: Raças": 17,
    "Capítulo 3: Classes": 45,
    "Capítulo 4: Personalidade e Antecedentes": 121,
    "Capítulo 5: Equipamento": 143,
    "Capítulo 6: Opções de Personalização": 163,
    "Capítulo 7: Utilizando Habilidades": 173,
    "Capítulo 8: Aventurando-se": 181,
    "Capítulo 9: Combate": 189,
    "Capítulo 10: Conjuração": 201,
    "Capítulo 11: Magias": 207,
    "Apêndice A: Condições": 290,
    "FIM": 999
}

console = Console(theme=Theme({"info": "cyan", "success": "bold green", "warning": "yellow"}))

SECTION_PATTERN = re.compile(r"(?:^|\n)\s*([A-ZÃÁÂÊÉÍÕÓÚÇ][A-ZÃÁÂÊÉÍÕÓÚÇ\s\-:]{3,})(?:\n|$)")

def clean_text(text):
    text = re.sub(r'LIVRO DO JOGADOR', '', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    return text

def process_pdf(file_path):
    console.print(Panel(f"📘 Lendo PDF (Modo PRO): {os.path.basename(file_path)}", style="blue"))
    loader = PyPDFLoader(file_path)
    all_pages = loader.load()
    total_pages_pdf = len(all_pages)
    
    rag_docs = []
    sorted_chapters = sorted(CHAPTER_MAP.items(), key=lambda x: x[1])

    for i in range(len(sorted_chapters) - 1):
        chapter_title, start_page = sorted_chapters[i]
        _, next_start_page = sorted_chapters[i+1]
        
        idx_start = max(0, start_page - 1 + PAGE_OFFSET)
        idx_end = min(total_pages_pdf, next_start_page - 1 + PAGE_OFFSET)

        if idx_start >= total_pages_pdf: continue

        # Extrai texto do capítulo
        chapter_pages = all_pages[idx_start:idx_end]
        chapter_text = "\n".join([p.page_content for p in chapter_pages])
        chapter_text = clean_text(chapter_text)

        console.print(f"[info]📖 Processando {chapter_title}...[/info]")

        # Divide por Sub-seções (Detector de Caixa Alta)
        sections = SECTION_PATTERN.split(chapter_text)
        current_section = "Introdução"

        for j, segment in enumerate(sections):
            segment = segment.strip()
            if not segment: continue

            # Se for título (impar na lista do split regex)
            if j % 2 != 0: 
                current_section = segment.title()
                continue
            
            # Processamento do Conteúdo
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_text(segment)
            
            for chunk in chunks:
                if len(chunk) < 50: continue
                
                # --- O SEGREDO DO RAG PRO ---
                # Injetamos o contexto explicitamente no texto que será vetorizado
                contextualized_content = f"Fonte: {chapter_title} > {current_section}\n---\n{chunk}"
                
                rag_docs.append({
                    "content": contextualized_content, # O texto rico vai aqui
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "chapter": chapter_title,
                        "section": current_section,
                        "original_content": chunk # Mantemos o original se precisar exibir limpo
                    }
                })

    return rag_docs

def main():
    if not os.path.exists(LIB_DIR):
        console.print(f"[error]Pasta {LIB_DIR} não existe![/error]")
        return

    pdf_files = [f for f in os.listdir(LIB_DIR) if f.lower().endswith(".pdf")]
    all_docs = []

    for pdf in pdf_files:
        docs = process_pdf(os.path.join(LIB_DIR, pdf))
        all_docs.extend(docs)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    console.print(Panel(f"💾 Sucesso! {len(all_docs)} chunks contextualizados gerados.\nSalvo em: {OUTPUT_JSON}\n[yellow]Agora rode o create_db_hybrid.py (aponte para este novo JSON)[/yellow]", style="bold green"))

if __name__ == "__main__":
    main()