import os
import json
import glob
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- CONFIGURAÇÃO ---
# Caminho para a pasta raiz que contém "2014" e "2024"
PASTA_RAIZ = "./5e-database-main" 

def carregar_json(caminho):
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Erro ao ler {caminho}: {e}")
        return []

def limpar_desc(desc_field):
    """Converte listas de strings em texto único."""
    if isinstance(desc_field, list):
        return "\n".join(desc_field)
    return str(desc_field or "")

def formatar_item(item, categoria_arquivo, ano):
    """
    Formata o item baseada na categoria (nome do arquivo).
    Adiciona o ano ao texto para o Mestre saber a versão da regra.
    """
    nome = item.get('name', 'Sem Nome')
    desc = limpar_desc(item.get('desc'))
    
    # Header padrão
    texto_base = f"SOURCE: 5e SRD ({ano})\nCATEGORY: {categoria_arquivo}\nITEM: {nome}\n"
    
    # 1. MAGIAS
    if "spells" in categoria_arquivo:
        detalhes = (f"LEVEL: {item.get('level', '?')}\n"
                    f"SCHOOL: {item.get('school', {}).get('name')}\n"
                    f"CASTING: {item.get('casting_time')}\nRANGE: {item.get('range')}\n"
                    f"COMPONENTS: {item.get('components', '')}\nDURATION: {item.get('duration')}")
        return f"{texto_base}{detalhes}\nDESC: {desc}"

    # 2. MONSTROS
    elif "monsters" in categoria_arquivo:
        actions = ", ".join([a['name'] for a in item.get('actions', [])])
        stats = (f"CR: {item.get('challenge_rating')}\n"
                 f"TYPE: {item.get('type')} ({item.get('size')})\n"
                 f"AC: {item.get('armor_class', [{}])[0].get('value', '?')}\nHP: {item.get('hit_points')}\n"
                 f"ACTIONS: {actions}")
        return f"{texto_base}{stats}\nSPECIAL: {limpar_desc(item.get('special_abilities', ''))}"

    # 3. CLASSES
    elif "classes" in categoria_arquivo and "subclasses" not in categoria_arquivo:
         return (f"{texto_base}HIT DIE: d{item.get('hit_die')}\n"
                 f"PROFICIENCIES: {', '.join([p.get('name') for p in item.get('proficiencies', [])])}\nDESC: {desc}")

    # 4. EQUIPAMENTO
    elif "equipment" in categoria_arquivo and "categories" not in categoria_arquivo:
        cost = item.get('cost', {})
        return f"{texto_base}COST: {cost.get('quantity')} {cost.get('unit')}\nDESC: {desc}"

    # 5. GENÉRICO (Todo o resto)
    else:
        # Tenta pegar atributos comuns de forma dinâmica
        extras = []
        for chave in ['type', 'alignment', 'hit_die']:
            if chave in item:
                val = item[chave]
                if isinstance(val, dict): val = val.get('name', val)
                extras.append(f"{chave.upper()}: {val}")
        return f"{texto_base}" + "\n".join(extras) + f"\nDESC: {desc}"

def processar_base_hibrida():
    # Dicionário para garantir a regra: "2024 sobrescreve 2014"
    # Chave: (categoria, nome_do_item) -> Valor: Documento
    consolidado = {}

    print(f"🚀 Iniciando processamento híbrido (2014 -> 2024)...")

    # A ordem importa! Processamos 2014 primeiro, depois 2024 sobrescreve.
    ordem_processamento = ["2014", "2024"]

    for ano in ordem_processamento:
        path_ano = os.path.join(PASTA_RAIZ, ano)
        if not os.path.exists(path_ano):
            print(f"⚠️ Pasta não encontrada: {path_ano}")
            continue

        arquivos = glob.glob(os.path.join(path_ano, "*.json"))
        print(f"📂 Processando {ano}: {len(arquivos)} arquivos encontrados.")

        for caminho_arquivo in arquivos:
            nome_arquivo = os.path.basename(caminho_arquivo)
            # Remove extensão e prefixo para criar uma categoria limpa
            categoria = nome_arquivo.lower().replace("5e-srd-", "").replace(".json", "")
            
            dados = carregar_json(caminho_arquivo)
            
            for item in dados:
                try:
                    nome_item = item.get('name')
                    if not nome_item: continue

                    # Cria o conteúdo do documento
                    texto_final = formatar_item(item, categoria, ano)
                    
                    # Cria a chave única para deduplicação
                    chave_unica = (categoria, nome_item)

                    # Cria o objeto Document
                    doc = Document(
                        page_content=texto_final,
                        metadata={
                            "source": categoria,
                            "name": nome_item,
                            "year": ano # Importante para saber a origem
                        }
                    )

                    # Adiciona ou Sobrescreve no dicionário
                    # Se for 2014 e já existir, adiciona.
                    # Se for 2024 e já existir (veio do 2014), isso vai substituir pelo novo!
                    consolidado[chave_unica] = doc
                
                except Exception as e:
                    continue
    
    # Converte o dicionário consolidado de volta para uma lista
    docs_finais = list(consolidado.values())

    # === REGRAS MANUAIS (CRIMINAL, ETC) ===
    # Elas entram por último e não são sobrescritas (ou podem ser, depende da lógica, aqui adicionamos)
    if os.path.exists("regras_extras.txt"):
        print("➕ Adicionando regras manuais...")
        with open("regras_extras.txt", "r", encoding="utf-8") as f:
            docs_finais.append(Document(
                page_content=f.read(), 
                metadata={"source": "manual", "name": "Custom Rules", "year": "Custom"}
            ))

    print(f"📦 Total Consolidado: {len(docs_finais)} documentos únicos.")
    return docs_finais

# --- EXECUÇÃO ---

print("🧠 Carregando Modelo de Embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

documentos = processar_base_hibrida()

if documentos:
    print("💾 Gravando no Banco Vetorial...")
    # Sugestão: Use uma nova pasta para testar
    persist_dir = "./dnd_db_hybrid_2024"
    
    # Se quiser limpar o banco antigo antes (Opcional, mas recomendado para evitar lixo)
    import shutil
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vector_db = Chroma.from_documents(
        documents=documentos,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    print(f"✅ SUCESSO! Banco híbrido criado em '{persist_dir}'")
    print("👉 Lembre de atualizar seu script de chat para apontar para essa nova pasta!")
else:
    print("❌ Nenhum dado processado.")