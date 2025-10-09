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
Analyze the data below and answer the user's question.

--- DATA ---
{data_csv}

--- QUESTION ---
{user_question}
"""
    return prompt

def generate_insight_huggingface(prompt):
    """
    Tenta usar o modelo 'gpt2' como um teste de diagnóstico definitivo.
    """
    model_id = "gpt2" # Usando o modelo mais básico e universal para teste
    model_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    st.info(f"A realizar um teste de diagnóstico com o modelo: {model_id}...")
    
    try:
        api_token = st.secrets["huggingface_api"]["token"]
        headers = {"Authorization": f"Bearer {api_token}"}
        
        payload = {
            "inputs": prompt,
            "options": {"wait_for_model": True}
        }
        
        response = requests.post(model_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            st.success(f"Modelo '{model_id}' respondeu com sucesso! A conexão está a funcionar.")
            result = response.json()
            generated_text = result[0]['generated_text']
            answer = generated_text.replace(prompt, "").strip()
            return answer
        else:
            # Mostra uma mensagem de erro muito mais detalhada
            st.error(f"O teste de diagnóstico falhou com o código {response.status_code}.")
            st.error(f"Resposta completa da API: {response.text}")
            st.warning("Isto indica um problema com o seu token de API nos 'Secrets' do Streamlit ou um bloqueio de rede. Por favor, verifique se o token foi copiado corretamente, sem espaços extra.")
            return None

    except Exception as e:
        st.error(f"Ocorreu uma exceção ao chamar a API: {e}")
        return None

# --- INTERFACE DO USUÁRIO (UI) ---

st.title("💡 Assistente de Análise de Colaborações")
st.markdown("Faça uma pergunta sobre as colaborações dos últimos 90 dias e a IA irá gerar um insight para você.")

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
                
                with st.spinner("A realizar teste de diagnóstico com a API do Hugging Face..."):
                    prompt = build_prompt(user_question, dados_df)
                    insight = generate_insight_huggingface(prompt)

                if insight:
                    st.subheader("Análise Gerada pela IA:")
                    st.markdown(insight)

