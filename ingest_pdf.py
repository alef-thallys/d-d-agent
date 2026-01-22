import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURAÇÃO ---
# Nome exato do arquivo PDF que você enviou
PDF_PATH = "dd-5e-livro-do-jogador-fundo-branco-biblioteca-elfica.pdf"
# A pasta do banco de dados que criamos no passo anterior
DB_DIR = "./dnd_db_full"

def processar_pdf():
    print(f"📘 Lendo o grimório: {PDF_PATH}...")
    
    if not os.path.exists(PDF_PATH):
        print("❌ Erro: Arquivo PDF não encontrado na pasta raiz.")
        return

    # 1. Carrega o PDF página por página
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"   > PDF carregado: {len(pages)} páginas encontradas.")

    # 2. Configura o "Cortador" de texto (Splitter)
    # chunk_size=1000: Cada pedaço terá ~1000 caracteres (aprox 1 parágrafo grande)
    # chunk_overlap=200: 200 caracteres repetidos entre pedaços para não cortar frases no meio
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""] # Tenta quebrar em parágrafos primeiro
    )

    # 3. Corta as páginas em pedaços menores (chunks)
    print("✂️  Dividindo o texto em chunks inteligentes...")
    chunks = text_splitter.split_documents(pages)
    
    # Adiciona metadados extras para o Mestre saber a origem
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "Players Handbook (PDF)"
        chunk.metadata["type"] = "pdf_content"
        # Opcional: Tentar limpar cabeçalhos repetitivos se necessário
        # chunk.page_content = chunk.page_content.replace("LIVRO DO JOGADOR", "")

    print(f"📦 Total de chunks criados: {len(chunks)}")
    return chunks

# --- EXECUÇÃO ---

print("🧠 Carregando Modelo de IA...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

novos_documentos = processar_pdf()

if novos_documentos:
    print(f"💾 Adicionando ao banco de dados existente em '{DB_DIR}'...")
    
    # Carrega o banco existente para ADICIONAR (não sobrescrever)
    vector_db = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=embedding_model
    )
    
    # Adiciona os novos chunks do PDF
    # O Chroma lida com o gerenciamento de IDs automaticamente
    vector_db.add_documents(novos_documentos)
    
    print("✅ SUCESSO! O Livro do Jogador foi assimilado pelo Mestre.")
else:
    print("❌ Nenhum texto foi processado.")