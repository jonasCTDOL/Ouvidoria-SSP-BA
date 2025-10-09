import streamlit as st
import mysql.connector
import pandas as pd
import google.generativeai as genai

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
        # Conecta ao banco usando as credenciais para MySQL salvas em st.secrets
        conn = mysql.connector.connect(**st.secrets["mysql"])

        # A query SQL é compatível com MySQL
        query = "SELECT * FROM colaboracoes WHERE created_at >= NOW() - INTERVAL 90 DAY;"

        # Usar o Pandas para ler o SQL diretamente é mais eficiente
        df = pd.read_sql(query, conn)
        
        conn.close()
        return df
        
    except mysql.connector.Error as e:
        st.error(f"Erro de Conexão com o Banco de Dados MySQL: {e}")
        # Mensagem de ajuda corrigida para 'database'
        st.info("Verifique se as credenciais no 'Secrets' (host, user, password, database) estão corretas e se o IP do Streamlit Cloud tem permissão de acesso remoto ao seu MySQL.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao buscar os dados: {e}")
        return None

def build_prompt(user_question, df):
    """Monta o prompt para o Gemini a partir da pergunta e dos dados."""
    data_csv = df.to_csv(index=False)
    
    prompt = f"""
    Você é um assistente de análise de dados especialista em segurança pública e colaboração cidadã.
    Sua tarefa é analisar os dados brutos em formato CSV fornecidos abaixo e responder à pergunta do usuário.
    Seja claro, objetivo e baseie sua resposta exclusivamente nos dados.

    --- DADOS BRUTOS (últimos 90 dias) ---
    {data_csv}

    --- PERGUNTA DO USUÁRIO ---
    Com base nos dados fornecidos, responda: {user_question}
    """
    return prompt

def generate_insight(prompt):
    """Envia o prompt para a API do Gemini e retorna a resposta."""
    try:
        genai.configure(api_key=st.secrets["google_api"]["key"])
        # CORREÇÃO: Alterado para 'gemini-pro' para máxima estabilidade e compatibilidade.
        model = genai.GenerativeModel('gemini-pro')
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Erro ao chamar a API do Gemini: {e}")
        return None

# --- INTERFACE DO USUÁRIO (UI) ---

st.title("💡 Assistente de Análise de Colaborações")
st.markdown("Faça uma pergunta em linguagem natural sobre as colaborações dos últimos 90 dias e a IA irá gerar um insight para você.")

default_question = "Qual cidade teve mais colaborações e qual o tipo de colaboração mais comum ('denuncia', 'sugestao', etc.)?"
user_question = st.text_area("Sua pergunta:", value=default_question, height=100)

if st.button("Gerar Insight"):
    if not user_question:
        st.warning("Por favor, digite uma pergunta para análise.")
    else:
        with st.spinner("Conectando ao banco de dados e buscando informações..."):
            dados_df = fetch_data_from_db()

        if dados_df is not None:
            if dados_df.empty:
                st.info("Nenhum registro de colaboração encontrado nos últimos 90 dias.")
            else:
                st.success(f"Dados carregados! {len(dados_df)} registros encontrados.")
                
                with st.spinner("A IA está pensando... Gerando seu insight agora."):
                    prompt = build_prompt(user_question, dados_df)
                    insight = generate_insight(prompt)

                if insight:
                    st.subheader("Análise Gerada pela IA:")
                    st.markdown(insight)

