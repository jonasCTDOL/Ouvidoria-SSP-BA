import streamlit as st
import mysql.connector
import pandas as pd
import requests # Usado para fazer chamadas à API do Hugging Face
import json

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análise de Colaborações com IA",
    page_icon="💡"
)

# --- FUNÇÕES DE LÓGICA ---

@st.cache_data(ttl=3600)
def fetch_data_from_db():
    """Conecta ao banco de dados MySQL usando os segredos e busca os dados."""
    try:
        conn = mysql.connector.connect(**st.secrets["mysql"])
        query = "SELECT * FROM colaboracoes WHERE created_at >= NOW() - INTERVAL 90 DAY;"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Ocorreu um erro ao conectar ou buscar dados: {e}")
        return None

def build_prompt(user_question, df):
    """Monta o prompt para um modelo de instrução a partir da pergunta e dos dados."""
    data_csv = df.to_csv(index=False)
    # Modelos de instrução funcionam melhor com um formato claro de tarefa
    prompt = f"""
[INST] Você é um assistente de análise de dados. Sua tarefa é analisar os dados em formato CSV abaixo e responder à pergunta do usuário de forma clara e concisa. Baseie sua resposta apenas nos dados fornecidos.

--- DADOS ---
{data_csv}

--- PERGUNTA ---
{user_question} [/INST]
"""
    return prompt

def generate_insight_huggingface(prompt):
    """
    Envia o prompt para a API do Hugging Face, tentando uma lista de modelos
    até encontrar um que responda com sucesso.
    """
    # Lista de modelos confiáveis para tentar em ordem de preferência.
    candidate_models = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "google/gemma-2-9b-it",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2"
    ]
    
    api_token = st.secrets["huggingface_api"]["token"]
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": prompt}

    for model_id in candidate_models:
        model_url = f"https://api-inference.huggingface.co/models/{model_id}"
        st.info(f"A testar o modelo de IA: {model_id}...")
        
        try:
            # Adicionado um timeout para evitar que a aplicação fique presa
            response = requests.post(model_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                st.success(f"Modelo '{model_id}' respondeu com sucesso!")
                result = response.json()
                generated_text = result[0]['generated_text']
                answer = generated_text.replace(prompt, "").strip()
                return answer
            elif response.status_code == 503: # Erro comum quando o modelo está a carregar
                 st.warning(f"O modelo '{model_id}' está a carregar. A tentar o próximo...")
                 continue
            else:
                 st.warning(f"O modelo '{model_id}' falhou com o código {response.status_code}. A tentar o próximo...")

        except requests.exceptions.RequestException as e:
            st.warning(f"Erro de rede ao tentar o modelo '{model_id}': {e}. A tentar o próximo...")
            continue
    
    # Se o loop terminar sem que nenhum modelo tenha respondido
    st.error("Não foi possível obter uma resposta de nenhum dos modelos de IA disponíveis. Por favor, tente novamente mais tarde.")
    return None

# --- INTERFACE DO USUÁRIO (UI) ---

st.title("💡 Assistente de Análise de Colaborações")
st.markdown("Faça uma pergunta sobre as colaborações dos últimos 90 dias e a IA irá gerar um insight para você.")
st.info("ℹ️ Esta demonstração usa modelos da comunidade Hugging Face. A primeira geração pode demorar um pouco mais enquanto o modelo é carregado.")

default_question = "Qual cidade teve mais colaborações e qual o tipo de colaboração mais comum ('denuncia', 'sugestao', etc.)?"
user_question = st.text_area("Sua pergunta:", value=default_question, height=100)

if st.button("Gerar Insight"):
    if not user_question:
        st.warning("Por favor, digite uma pergunta para análise.")
    else:
        with st.spinner("Conectando ao banco de dados..."):
            dados_df = fetch_data_from_db()

        if dados_df is not None:
            if dados_df.empty:
                st.info("Nenhum registro encontrado nos últimos 90 dias.")
            else:
                st.success(f"Dados carregados! {len(dados_df)} registros encontrados.")
                
                # O spinner agora é mais genérico, pois a função interna dará o feedback
                with st.spinner("A contactar os modelos de IA do Hugging Face..."):
                    prompt = build_prompt(user_question, dados_df)
                    insight = generate_insight_huggingface(prompt)

                if insight:
                    st.subheader("Análise Gerada pela IA:")
                    st.markdown(insight)

