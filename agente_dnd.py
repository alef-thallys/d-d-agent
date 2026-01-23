import os
import time
from dotenv import load_dotenv

# --- IMPORTS DO LANGCHAIN (LCEL) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory

# --- INTERFACE VISUAL (RICH) ---
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live

# --------------------------------------------------
# CONFIGURAÇÃO INICIAL
# --------------------------------------------------
load_dotenv()
console = Console()

DB_DIR = "./dnd_db_2026"
GEMINI_MODEL = "gemini-flash-latest"

# --------------------------------------------------
# CHECAGEM DE SISTEMA
# --------------------------------------------------
def check_system():
    with console.status(
        "[bold yellow]🔍 Verificando integridade dos grimórios...[/bold yellow]",
        spinner="dots"
    ):
        time.sleep(1)

        if "GOOGLE_API_KEY" not in os.environ:
            console.print("[bold red]❌ GOOGLE_API_KEY não encontrada[/bold red]")
            console.print("[yellow]→ Verifique o arquivo .env[/yellow]")
            raise SystemExit(1)

        if not os.path.exists(DB_DIR):
            console.print(f"[bold red]❌ Banco vetorial '{DB_DIR}' não encontrado[/bold red]")
            console.print("[yellow]→ Execute create_db_hybrid.py ou ingest_pdf.py primeiro[/yellow]")
            raise SystemExit(1)

    console.print("[bold green]✅ Sistema validado![/bold green]\n")

# --------------------------------------------------
# PROMPT (LCEL) COM MEMÓRIA
# --------------------------------------------------
PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um Mestre de D&D 5ª Edição experiente.\n"
        "Use o contexto fornecido como FONTE PRINCIPAL.\n"
        "Quando a resposta não estiver explícita, mas puder ser deduzida pelas regras oficiais de D&D 5e, explique o raciocínio.\n"
        "Não invente magias ou regras inexistentes.\n"
        "Responda em português usando Markdown."
    ),
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        "CONTEXTO DO GRIMÓRIO (REGRAS):\n{context}\n\n"
        "PERGUNTA DO JOGADOR:\n{question}"
    )
])


# --------------------------------------------------
# SETUP DO AGENTE (LCEL MODERNO)
# --------------------------------------------------
def setup_agent():
    with console.status(
        "[bold purple]🔮 Invocando o Mestre dos Magos...[/bold purple]",
        spinner="moon"
    ):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # ⚠️ CORREÇÃO: Adicionado collection_name para achar os dados do PDF
        vector_db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name="dnd_rules" 
        )

        retriever = vector_db.as_retriever(search_kwargs={"k": 8})

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.0,
            streaming=False
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain LCEL com injeção de contexto + memória
        rag_chain = (
            RunnablePassthrough.assign(
                context=(lambda x: x["question"]) | retriever | format_docs
            )
            | PROMPT
            | llm
            | StrOutputParser()
        )

        # Histórico em memória volátil
        chat_history = ChatMessageHistory()

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: chat_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    return conversational_chain

# --------------------------------------------------
# LOOP PRINCIPAL
# --------------------------------------------------
def main():
    console.clear()
    console.print(Panel.fit(
        "[bold yellow]🐉 RAG D&D 5e — Mestre Digital (LCEL)[/bold yellow]\n"
        "[italic]Pergunte sobre regras, magias, monstros...[/italic]",
        border_style="red",
        subtitle="v3.3 • Full Connected"
    ))

    check_system()
    chain = setup_agent()

    console.print("[dim]Sistema online. Digite 'sair' para encerrar.[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]🧙 Você:[/bold cyan] ")

            if user_input.lower() in {"sair", "exit", "quit"}:
                console.print("[bold red]🎲 O Mestre encerra a sessão.[/bold red]")
                break

            if not user_input.strip():
                continue

            start_time = time.time()
            resposta_final = ""

            with Live(
                Panel("Consultando os grimórios...", title="📜 Mestre", border_style="green"),
                refresh_per_second=12
            ) as live_panel:

                resposta = chain.invoke(
                    {"question": user_input},
                    config={"configurable": {"session_id": "mesa-principal"}}
                )

                # Renderiza resposta completa corretamente em Markdown
                live_panel.update(
                    Panel(
                        Markdown(resposta),
                        title="📜 Mestre",
                        border_style="green",
                        padding=(1, 2)
                    )
                )

            tempo = time.time() - start_time
            console.print(f"[dim right]Tempo: {tempo:.2f}s[/dim right]\n")

        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            console.print(f"[bold red]❌ Erro inesperado:[/bold red] {e}")

if __name__ == "__main__":
    main()