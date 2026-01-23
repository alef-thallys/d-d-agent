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
OUTPUT_JSON = "rag_ready_manual.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# A Página 11 do livro (Início Cap 1) é a Página 10 do arquivo PDF.
# Portanto, 11 + OFFSET = 10  =>  OFFSET = -1
PAGE_OFFSET = -1

# --- MAPA DE CAPÍTULOS (NOME -> PÁGINA INICIAL) ---
# O script vai ler da página X até o início da próxima.
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
    "FIM": 999  # Marcador para saber onde termina o último capítulo
}

console = Console(theme=Theme({"info": "cyan", "success": "bold green", "warning": "yellow"}))

# Regex apenas para Sub-seções (ex: "ANÃO", "MAGIA", "COMBATE")
SECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*([A-ZÃÁÂÊÉÍÕÓÚÇ][A-ZÃÁÂÊÉÍÕÓÚÇ\s\-:]{3,})(?:\n|$)"
)

def clean_text(text):
    # Remove cabeçalhos e números de página soltos
    text = re.sub(r'LIVRO DO JOGADOR', '', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text) # Remove numeração isolada
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text) # Corrige hífens
    # Remove números que aparecem sozinhos logo no início do texto
    text = re.sub(r'^\s*\d+\s*\n', '', text) 
    return text

def process_pdf(file_path):
    console.print(Panel(f"📘 Lendo PDF: {os.path.basename(file_path)}", style="blue"))
    
    loader = PyPDFLoader(file_path)
    # Carrega todas as páginas de uma vez (pode demorar um pouco se for gigante)
    all_pages = loader.load()
    total_pages_pdf = len(all_pages)
    
    rag_docs = []
    
    # Ordena os capítulos pela página para garantir a sequência
    sorted_chapters = sorted(CHAPTER_MAP.items(), key=lambda x: x[1])

    for i in range(len(sorted_chapters) - 1):
        chapter_title, start_page = sorted_chapters[i]
        _, next_start_page = sorted_chapters[i+1]
        
        # Ajusta para índice do Python (0-based) e aplica Offset
        idx_start = max(0, start_page - 1 + PAGE_OFFSET)
        idx_end = min(total_pages_pdf, next_start_page - 1 + PAGE_OFFSET)

        if idx_start >= total_pages_pdf:
            console.print(f"[warning]⚠️ Capítulo '{chapter_title}' começa na pág {start_page}, mas o PDF só tem {total_pages_pdf} págs.[/warning]")
            continue

        # Extrai o texto desse intervalo de páginas
        chapter_pages = all_pages[idx_start:idx_end]
        chapter_text = "\n".join([p.page_content for p in chapter_pages])
        chapter_text = clean_text(chapter_text)

        console.print(f"[info]📖 Processando {chapter_title} (Págs {start_page}-{next_start_page-1})[/info]")

        # --- DIVISÃO POR SEÇÕES (IGUAL ANTES) ---
        sections = SECTION_PATTERN.split(chapter_text)
        current_section = "Geral"

        for j, segment in enumerate(sections):
            segment = segment.strip()
            if not segment: continue

            # Verifica se é Título de Seção (Caixa Alta e curto)
            if j % 2 != 0 and len(segment) < 100:
                current_section = segment.title()
                continue
            
            # É conteúdo
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_text(segment)
            
            for chunk in chunks:
                if len(chunk) < 50: continue
                rag_docs.append({
                    "content": chunk,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "chapter": chapter_title,
                        "section": current_section,
                        "page_range": f"{start_page}-{next_start_page-1}"
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

    console.print(Panel(f"💾 Sucesso! {len(all_docs)} chunks gerados manualmente.\nSalvo em: {OUTPUT_JSON}", style="bold green"))

if __name__ == "__main__":
    main()