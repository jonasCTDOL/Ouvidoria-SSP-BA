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
    e fornecendo diagnósticos de erro detalhados.
    """
    candidate_models = [
        "HuggingFaceH4/zephyr-7b-beta",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "google/gemma-2-9b-it",
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ]
    
    try:
        api_token = st.secrets["huggingface_api"]["token"]
        headers = {"Authorization": f"Bearer {api_token}"}
        # --- NOVO DIAGNÓSTICO ---
        st.info(f"A usar o token que começa com '{api_token[:6]}' e termina com '{api_token[-4:]}'. Verifique se corresponde ao seu token no site do Hugging Face.")
        # --- FIM DO DIAGNÓSTICO ---
    except Exception as e:
        st.error("Erro ao ler o token da API. Verifique se a secção `[huggingface_api]` com a chave `token` existe nos seus 'Secrets' do Streamlit.")
        return None

    for model_id in candidate_models:
        model_url = f"https://api-inference.huggingface.co/models/{model_id}"
        st.info(f"A testar o modelo de IA: {model_id}...")
        
        try:
            payload = {
                "inputs": prompt,
                "options": {"wait_for_model": True}
            }
            response = requests.post(model_url, headers=headers, json=payload, timeout=90)
            
            if response.status_code == 200:
                st.success(f"Modelo '{model_id}' respondeu com sucesso!")
                result = response.json()
                generated_text = result[0]['generated_text']
                answer = generated_text.replace(prompt, "").strip()
                return answer
            elif response.status_code == 401:
                st.error("Erro de Autenticação (401). O seu token de API do Hugging Face é inválido ou foi copiado incorretamente. Por favor, gere um novo token, copie-o com cuidado e atualize os 'Secrets' no Streamlit.")
                return None # Para de tentar se a autenticação falhar
            else:
                 st.warning(f"O modelo '{model_id}' falhou com o código {response.status_code}. Resposta: {response.text}. A tentar o próximo modelo...")

        except requests.exceptions.RequestException as e:
            st.warning(f"Erro de rede ao tentar o modelo '{model_id}': {e}. A tentar o próximo...")
            continue
    
    st.error("Não foi possível obter uma resposta de nenhum dos modelos de IA disponíveis. Por favor, tente novamente mais tarde.")
    return None

# --- INTERFACE DO USUÁRIO (UI) ---

st.title("💡 Assistente de Análise de Colaborações")
st.markdown("Faça uma pergunta sobre as colaborações dos últimos 90 dias e a IA irá gerar um insight para você.")
st.info("ℹ️ Esta demonstração usa modelos da comunidade Hugging Face. A primeira geração pode demorar mais enquanto o modelo é carregado.")

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
                
                with st.spinner("A contactar os modelos de IA do Hugging Face..."):
                    prompt = build_prompt(user_question, dados_df)
                    insight = generate_insight_huggingface(prompt)

                if insight:
                    st.subheader("Análise Gerada pela IA:")
                    st.markdown(insight)

