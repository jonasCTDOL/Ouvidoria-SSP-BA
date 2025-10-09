import streamlit as st
import mysql.connector
import pandas as pd
import google.generativeai as genai

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análise de Colaborações com IA",
    page_icon="💡"
)

# --- FUNÇÕES DE LÓgica ---

@st.cache_data(ttl=3600)
def fetch_data_from_db():
    """Conecta ao banco de dados MySQL usando os segredos e busca os dados."""
    try:
        conn = mysql.connector.connect(**st.secrets["mysql"])
        query = "SELECT * FROM colaboracoes WHERE created_at >= NOW() - INTERVAL 90 DAY;"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except mysql.connector.Error as e:
        st.error(f"Erro de Conexão com o Banco de Dados MySQL: {e}")
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
        # CORREÇÃO 2: Usando o nome de modelo versionado 'gemini-1.0-pro' para máxima especificidade.
        model = genai.GenerativeModel('gemini-1.0-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Erro ao chamar a API do Gemini: {e}")
        return None

# --- Ferramenta de Diagnóstico (NOVA) ---
def list_available_models():
    """Lista os modelos disponíveis para a chave de API configurada."""
    try:
        genai.configure(api_key=st.secrets["google_api"]["key"])
        # Filtra para mostrar apenas modelos que podem gerar conteúdo, que é o que precisamos.
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return models
    except Exception as e:
        return [f"Erro ao tentar listar os modelos: {e}"]

# --- INTERFACE DO USUÁRIO (UI) ---

st.title("💡 Assistente de Análise de Colaborações")
st.markdown("Faça uma pergunta em linguagem natural sobre as colaborações dos últimos 90 dias e a IA irá gerar um insight para você.")

# Seção principal da aplicação
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

st.divider()

# Seção de diagnóstico expansível
with st.expander("🔬 Ferramentas de Diagnóstico da API"):
    st.write("Se estiver enfrentando erros com a API do Gemini, use esta ferramenta para verificar sua conexão.")
    if st.button("Listar Modelos Disponíveis"):
        with st.spinner("Consultando a API do Google para ver os modelos que sua chave pode acessar..."):
            model_list = list_available_models()
            st.write("Modelos que sua API Key pode usar para gerar conteúdo:")
            st.json(model_list)
            st.info("Se a lista estiver vazia ou mostrar um erro, verifique se a API 'Generative Language' e o Faturamento estão ativos no seu projeto Google Cloud.")

