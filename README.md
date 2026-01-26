# 🎲 D&D Agent - Assistente Inteligente para Dungeons & Dragons 5ª Edição

Um agente conversacional RAG (Retrieval-Augmented Generation) especializado em regras de D&D 5e, usando busca híbrida (vetorial + BM25) e modelo Gemini para fornecer respostas precisas baseadas nos livros de D&D.

## ✨ Características

- 🔮 **Busca Híbrida**: Combina busca vetorial (Chroma) e BM25 para recuperação de informações mais precisa
- 🤖 **Multi-Query Retriever**: Reformula perguntas para melhorar a busca
- 💬 **Memória de Conversa**: Mantém contexto da sessão para diálogos naturais
- 🎯 **Respostas Precisas**: Apenas responde com base no conteúdo dos PDFs (sem alucinações)
- 🌐 **100% Local**: Embeddings processados localmente, apenas o LLM usa API

## 🛠️ Tecnologias

- **LangChain**: Framework para aplicações LLM
- **Google Gemini**: Modelo de linguagem (gemini-flash-latest)
- **ChromaDB**: Banco de dados vetorial
- **Sentence Transformers**: Embeddings multilíngues locais
- **BM25**: Busca lexical tradicional
- **Rich**: Interface de terminal elegante

## 📋 Pré-requisitos

- Python 3.9+
- Chave de API do Google Gemini ([obter aqui](https://aistudio.google.com/app/apikey))
- 4GB+ de RAM (para processar embeddings)

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/alef-thallys/dnd-grimoire.git
cd dnd-grimoire
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure a API Key

```bash
# Copie o arquivo de exemplo
cp .env-example .env

# Edite o arquivo .env e adicione sua chave
# GOOGLE_API_KEY=sua_chave_aqui
```

## 📖 Como Usar

O projeto já inclui o PDF do **Livro do Jogador D&D 5e** na pasta `biblioteca/`. Siga os passos abaixo:

### **Passo 1: Processar o PDF**

Este script extrai e limpa o texto do PDF, organizando por capítulos:

```bash
python ingest_pdf.py
```

**O que faz:**
- Lê o PDF da pasta `biblioteca/`
- Divide por capítulos conforme mapeamento interno
- Limpa cabeçalhos, rodapés e hifenização
- Gera o arquivo `rag_v5.json` com fragmentos estruturados

**Saída esperada:** ✅ Arquivo `rag_v5.json` criado (~1000+ fragmentos)

---

### **Passo 2: Criar Banco Vetorial**

Transforma o JSON em um banco de dados vetorial local:

```bash
python create_db_hybrid.py
```

**O que faz:**
- Carrega `rag_v5.json`
- Gera embeddings usando Sentence Transformers (modelo multilíngue)
- Cria banco ChromaDB na pasta `./dnd_db_2026`

**Tempo estimado:** 2-5 minutos (depende do processador)

**Saída esperada:** ✅ Pasta `dnd_db_2026` criada com o banco

---

### **Passo 3: Conversar com o Agente**

Inicie o chat interativo:

```bash
python agente_dnd.py
```

**Exemplos de perguntas:**
```
📜 Como funciona o ataque furtivo do ladino?
📜 Quais são as características do guerreiro?
📜 Como calcular CA?
📜 Qual a diferença entre ação e ação bônus?
```

**Para sair:** Digite `sair`, `exit` ou `quit`

---

## 📁 Estrutura do Projeto

```
dnd-grimoire/
├── biblioteca/                      # PDFs de D&D (já incluído)
│   └── dd-5e-livro-do-jogador-fundo-branco-biblioteca-elfica.pdf
├── agente_dnd.py                    # Script principal do agente
├── ingest_pdf.py                    # Processa PDF → JSON
├── create_db_hybrid.py              # Cria banco vetorial
├── requirements.txt                 # Dependências Python
├── .env-example                     # Exemplo de configuração
├── rag_v5.json                      # Dados processados (gerado)
└── dnd_db_2026/                     # Banco vetorial (gerado)
```

## ⚙️ Configurações Avançadas

### Ajustar Modelo Gemini

Em `agente_dnd.py`, linha 21:

```python
GEMINI_MODEL = "gemini-flash-latest"  # Rápido e econômico
# GEMINI_MODEL = "gemini-1.5-pro"     # Mais preciso
```

### Modificar Chunking

Em `ingest_pdf.py`, linhas 10-11:

```python
CHUNK_SIZE = 1000        # Tamanho dos fragmentos
CHUNK_OVERLAP = 200      # Sobreposição entre chunks
```

### Ajustar Pesos da Busca Híbrida

Em `agente_dnd.py`, linha 47:

```python
weights=[0.6, 0.4]  # [Vetorial, BM25]
# Aumente 0.6 para priorizar similaridade semântica
# Aumente 0.4 para priorizar correspondência exata de termos
```

## 🧠 Como Funciona (RAG Híbrido)

1. **Ingestão**: PDF → Chunks com metadados (capítulo/seção)
2. **Indexação**: 
   - Embeddings vetoriais (Sentence Transformers)
   - Índice BM25 (busca por palavras-chave)
3. **Retrieval**:
   - Query → Multi-Query (gera variações da pergunta)
   - Busca híbrida (60% vetorial + 40% BM25)
   - Retorna top 4 documentos mais relevantes
4. **Geração**:
   - Contexto + Histórico → Gemini
   - Resposta fundamentada apenas no contexto recuperado

## 🐛 Troubleshooting

### Erro: "JSON não encontrado"
```bash
# Execute primeiro:
python ingest_pdf.py
```

### Erro: "API Key inválida"
```bash
# Verifique se o .env está configurado:
cat .env  # Linux/Mac
type .env  # Windows
```

### Embeddings muito lentos
- Normal em primeira execução (baixa modelo ~400MB)
- Reduza `CHUNK_SIZE` no `ingest_pdf.py`
- Use CPU com AVX2 para acelerar

### Banco de dados corrompido
```bash
# Delete e recrie:
rm -rf dnd_db_2026  # Linux/Mac
rmdir /s dnd_db_2026  # Windows

python create_db_hybrid.py
```

## 📜 Licença

Este projeto é um estudo acadêmico. Os PDFs de D&D são propriedade da Wizards of the Coast.

## 🤝 Contribuições

Sugestões e melhorias são bem-vindas! Abra uma issue ou PR.

---

**Feito com ❤️ e um d20 de sorte**
