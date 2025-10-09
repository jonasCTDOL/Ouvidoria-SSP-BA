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

def list_available_models():
    """Lista os modelos disponíveis que a chave de API pode usar."""
    try:
        genai.configure(api_key=st.secrets["google_api"]["key"])
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return models
    except Exception as e:
        # Retorna o erro como o único item da lista para tratamento
        return [f"Erro ao listar modelos: {e}"]

def generate_insight(prompt, model_name):
    """Envia o prompt para a API do Gemini usando um nome de modelo específico."""
    try:
        genai.configure(api_key=st.secrets["google_api"]["key"])
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Erro ao chamar a API do Gemini com o modelo '{model_name}': {e}")
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
        # Etapa 1: Buscar os dados do banco
        with st.spinner("Conectando ao banco de dados..."):
            dados_df = fetch_data_from_db()

        if dados_df is not None:
            if dados_df.empty:
                st.info("Nenhum registro de colaboração encontrado nos últimos 90 dias.")
            else:
                st.success(f"Dados carregados! {len(dados_df)} registros encontrados.")
                
                # Etapa 2: Verificar quais modelos a API Key pode usar
                with st.spinner("Verificando permissões da API Key..."):
                    available_models = list_available_models()

                # Etapa 3: Tentar gerar o insight se houver modelos disponíveis
                if available_models and not available_models[0].startswith("Erro"):
                    model_to_use = available_models[0]
                    st.info(f"Usando o primeiro modelo disponível: `{model_to_use}`")

                    with st.spinner("A IA está pensando... Gerando seu insight agora."):
                        prompt = build_prompt(user_question, dados_df)
                        insight = generate_insight(prompt, model_to_use)

                    if insight:
                        st.subheader("Análise Gerada pela IA:")
                        st.markdown(insight)
                else:
                    # Se não houver modelos ou ocorrer um erro, exibir mensagem detalhada
                    st.error("**Falha na verificação da API do Google Gemini!**")
                    st.write("Sua chave de API não conseguiu listar os modelos disponíveis.")
                    st.write("**Causa Provável:**")
                    st.markdown("""
                    1.  A **API "Generative Language"** (ou Vertex AI) não está **ATIVADA** no seu projeto Google Cloud.
                    2.  O **Faturamento (Billing)** não está **ATIVO** para este projeto.
                    """)
                    st.write("**O que fazer:**")
                    st.markdown("""
                    1.  Acesse o [Google Cloud Console](https://console.cloud.google.com/).
                    2.  Verifique se o faturamento e a API correta estão ativados para o projeto associado à sua chave.
                    """)
                    if available_models:
                        st.write("Detalhes do erro retornado pela API:")
                        st.code(available_models[0])

