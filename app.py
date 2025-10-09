import streamlit as st
import mysql.connector
import pandas as pd
import requests
from huggingface_hub import InferenceClient # Importa o cliente oficial do Hugging Face

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análise de Colaborações com IA",
    page_icon="💡"
)

# --- FUNÇÕES DE LÓGICA ---

@st.cache_data(ttl=3600)
def fetch_data_from_db():
    """Conecta-se à base de dados MySQL usando os segredos e busca os dados."""
    try:
        conn = mysql.connector.connect(**st.secrets["mysql"])
        query = "SELECT * FROM colaboracoes WHERE created_at >= NOW() - INTERVAL 90 DAY;"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Ocorreu um erro ao conectar-se ou buscar dados: {e}")
        return None

def build_prompt(user_question, df):
    """Monta o prompt para um modelo de instrução a partir da pergunta e dos dados."""
    data_csv = df.to_csv(index=False)
    # Formato de prompt mais genérico para ser compatível com vários modelos
    prompt = f"""Task: You are a data analysis assistant. Your task is to analyze the CSV data provided below and answer the user's question clearly and concisely. Base your answer only on the data provided.

--- Data ---
{data_csv}

--- Question ---
{user_question}

--- Answer ---
"""
    return prompt

def generate_insight_huggingface(prompt):
    """
    Envia o prompt para a API do Hugging Face usando a biblioteca oficial 'huggingface_hub',
    tentando uma lista de modelos e fornecendo diagnósticos de erro detalhados.
    """
    candidate_models = [
        "HuggingFaceH4/zephyr-7b-beta",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "google/gemma-2-9b-it",
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ]
    
    try:
        api_token = st.secrets["huggingface_api"]["token"]
        st.info(f"A usar o token que começa com '{api_token[:6]}' e termina com '{api_token[-4:]}'. Verifique se corresponde ao seu token no site do Hugging Face.")
    except Exception as e:
        st.error("Erro ao ler o token da API. Verifique se a secção `[huggingface_api]` com a chave `token` existe nos seus 'Secrets' do Streamlit.")
        return None

    for model_id in candidate_models:
        st.info(f"A testar o modelo de IA: {model_id}...")
        
        try:
            # Nova abordagem: Usando o cliente oficial da biblioteca
            client = InferenceClient(model=model_id, token=api_token)
            
            # O método text_generation é o correto para este tipo de modelo
            response_text = client.text_generation(prompt, max_new_tokens=512)
            
            st.success(f"Modelo '{model_id}' respondeu com sucesso!")
            return response_text.strip()

        except Exception as e:
            error_message = str(e)
            if "401" in error_message:
                 st.error(f"Erro de Autenticação (401) com o modelo '{model_id}'. O seu token de API do Hugging Face é inválido. Por favor, verifique se o copiou corretamente para os 'Secrets' no Streamlit.")
                 return None # Para de tentar se a autenticação falhar
            else:
                 st.warning(f"O modelo '{model_id}' falhou com um erro: {error_message}. A tentar o próximo modelo...")
            continue
    
    st.error("Não foi possível obter uma resposta de nenhum dos modelos de IA disponíveis. Por favor, tente novamente mais tarde.")
    return None

# --- INTERFACE DO UTILIZADOR (UI) ---

st.title("💡 Assistente de Análise de Colaborações")
st.markdown("Faça uma pergunta sobre as colaborações dos últimos 90 dias e a IA irá gerar um insight para si.")
st.info("ℹ️ Esta demonstração usa modelos da comunidade Hugging Face. A primeira geração pode demorar mais enquanto o modelo é carregado.")

default_question = "Qual cidade teve mais colaborações e qual o tipo de colaboração mais comum ('denuncia', 'sugestao', etc.)?"
user_question = st.text_area("A sua pergunta:", value=default_question, height=100)

if st.button("Gerar Insight"):
    if not user_question:
        st.warning("Por favor, digite uma pergunta para análise.")
    else:
        # Ação Final: Verificação do Token
        # Recomendo que verifique uma última vez se o token nos Secrets está correto.
        # Um erro ao copiar e colar é a causa mais comum destes problemas.
        
        with st.spinner("A conectar-se à base de dados..."):
            dados_df = fetch_data_from_db()

        if dados_df is not None:
            if dados_df.empty:
                st.info("Nenhum registo encontrado nos últimos 90 dias.")
            else:
                st.success(f"Dados carregados! {len(dados_df)} registos encontrados.")
                
                with st.spinner("A contactar os modelos de IA do Hugging Face..."):
                    prompt = build_prompt(user_question, dados_df)
                    insight = generate_insight_huggingface(prompt)

                if insight:
                    st.subheader("Análise Gerada pela IA:")
                    st.markdown(insight)

