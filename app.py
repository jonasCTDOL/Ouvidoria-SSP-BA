import streamlit as st
import mysql.connector
import pandas as pd
from huggingface_hub import InferenceClient

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análise de Colaborações com IA",
    page_icon="💡"
)

# --- FUNÇÕES DE LÓGICA ---

@st.cache_data(ttl=3600)
def fetch_data_from_db():
    """Conecta-se à base de dados MySQL e busca todos os dados."""
    try:
        conn = mysql.connector.connect(**st.secrets["mysql"])
        query = "SELECT * FROM colaboracoes;"
        df = pd.read_sql(query, conn)
        conn.close()
        # Converte colunas de data para um formato mais legível
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at']).dt.date
        return df
    except Exception as e:
        st.error(f"Ocorreu um erro ao conectar-se ou buscar dados: {e}")
        return None

def get_data_summary(df):
    """Cria um resumo estatístico e informativo do DataFrame."""
    summary_parts = []
    
    summary_parts.append(f"Resumo Geral do Conjunto de Dados:")
    summary_parts.append(f"- Número total de colaborações: {len(df)}")
    
    if 'created_at' in df.columns and not df['created_at'].empty:
        summary_parts.append(f"- Período dos dados: de {df['created_at'].min()} a {df['created_at'].max()}")
        
    summary_parts.append("\nAnálise das Colunas Principais:")
    
    # Descreve as colunas mais importantes de forma legível
    for col in ['tipo_colaboracao', 'cidade', 'estado', 'status', 'anonimato']:
        if col in df.columns:
            summary_parts.append(f"\n- Contagem de valores para a coluna '{col}':")
            # Mostra a contagem dos valores mais comuns
            value_counts = df[col].value_counts().to_string()
            summary_parts.append(value_counts)
            
    return "\n".join(summary_parts)

def generate_insight_huggingface(user_question, df):
    """
    Envia um resumo dos dados para a API do Hugging Face para obter uma resposta inteligente.
    """
    candidate_models = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "google/gemma-2-9b-it",
        "HuggingFaceH4/zephyr-7b-beta",
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ]
    
    try:
        api_token = st.secrets["huggingface_api"]["token"]
        st.info(f"A usar o token que começa com '{api_token[:6]}' e termina com '{api_token[-4:]}'.")
    except Exception as e:
        st.error("Erro ao ler o token da API. Verifique a secção `[huggingface_api]` nos seus 'Secrets'.")
        return None

    # NOVIDADE: Gera um resumo inteligente em vez de usar os dados brutos.
    data_summary = get_data_summary(df)

    system_prompt = """Você é um analista de dados de elite. A sua única função é analisar o resumo estatístico que o utilizador fornece e responder à pergunta dele de forma clara e profissional, em português. Baseie a sua resposta **exclusivamente** no resumo. Não invente informações."""

    user_prompt = f"""
Aqui está um resumo estatístico dos dados de colaborações:
--- RESUMO DOS DADOS ---
{data_summary}

--- PERGUNTA DO UTILIZADOR ---
Com base **exclusivamente** no resumo acima, responda à seguinte pergunta: {user_question}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for model_id in candidate_models:
        st.info(f"A testar o modelo de IA: {model_id}...")
        
        try:
            client = InferenceClient(model=model_id, token=api_token)
            response = client.chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.3, # Diminuído para respostas ainda mais factuais
                top_p=0.95
            )
            
            insight = response.choices[0].message.content
            
            st.success(f"Modelo '{model_id}' respondeu com sucesso!")
            return insight.strip()

        except Exception as e:
            error_message = str(e)
            if "401" in error_message:
                 st.error(f"Erro de Autenticação (401) com o modelo '{model_id}'. O seu token de API é inválido.")
                 return None
            else:
                 st.warning(f"O modelo '{model_id}' falhou com um erro: {error_message}. A tentar o próximo modelo...")
            continue
    
    st.error("Não foi possível obter uma resposta de nenhum dos modelos de IA disponíveis. Por favor, tente novamente mais tarde.")
    return None

# --- INTERFACE DO UTILIZADOR (UI) ---

st.title("💡 Assistente de Análise de Colaborações")
st.markdown("Faça uma pergunta sobre o **histórico completo** de colaborações e a IA irá gerar um insight para si.")
st.info("ℹ️ A aplicação faz agora uma pré-análise inteligente dos dados para respostas mais precisas.")

default_question = "Qual cidade teve mais colaborações e qual o tipo de colaboração mais comum?"
user_question = st.text_area("A sua pergunta:", value=default_question, height=100)

if st.button("Gerar Insight"):
    if not user_question:
        st.warning("Por favor, digite uma pergunta para análise.")
    else:
        with st.spinner("A conectar-se à base de dados e a buscar todo o histórico..."):
            dados_df = fetch_data_from_db()

        if dados_df is not None:
            if dados_df.empty:
                st.info("Nenhum registo encontrado na base de dados.")
            else:
                st.success(f"Dados carregados! {len(dados_df)} registos encontrados.")
                
                with st.spinner("O pré-analista inteligente está a resumir os dados e a contactar a IA..."):
                    insight = generate_insight_huggingface(user_question, dados_df)

                if insight:
                    st.subheader("Análise Gerada pela IA:")
                    st.markdown(insight)

