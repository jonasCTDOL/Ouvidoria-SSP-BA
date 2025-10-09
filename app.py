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
    """Envia o prompt para a API do Hugging Face e retorna a resposta."""
    # CORREÇÃO: Trocado para um modelo alternativo e robusto da MistralAI.
    model_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    try:
        api_token = st.secrets["huggingface_api"]["token"]
        headers = {"Authorization": f"Bearer {api_token}"}
        
        payload = {"inputs": prompt}
        
        response = requests.post(model_url, headers=headers, json=payload)
        
        # Verifica se a resposta foi bem-sucedida
        if response.status_code == 200:
            # A resposta da API do Hugging Face vem numa lista
            result = response.json()
            # Pega o texto gerado do primeiro (e único) resultado
            generated_text = result[0]['generated_text']
            # O modelo repete o prompt na resposta, então removemos o prompt original
            # para obter apenas a resposta da IA.
            answer = generated_text.replace(prompt, "").strip()
            return answer
        else:
            st.error(f"Erro ao chamar a API do Hugging Face: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        st.error(f"Ocorreu uma exceção ao chamar a API do Hugging Face: {e}")
        return None

# --- INTERFACE DO USUÁRIO (UI) ---

st.title("💡 Assistente de Análise de Colaborações")
st.markdown("Faça uma pergunta sobre as colaborações dos últimos 90 dias e a IA irá gerar um insight para você.")
st.info("ℹ️ Esta demonstração usa um modelo da comunidade Hugging Face. A primeira geração pode demorar um pouco mais.")

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
                
                with st.spinner("A IA do Hugging Face está a processar... (pode demorar um pouco na primeira vez)"):
                    prompt = build_prompt(user_question, dados_df)
                    insight = generate_insight_huggingface(prompt)

                if insight:
                    st.subheader("Análise Gerada pela IA:")
                    st.markdown(insight)

