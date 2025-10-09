import streamlit as st
import psycopg2
import google.generativeai as genai
import pandas as pd

# --- CONFIGURAÇÃO DA PÁGINA ---
# É uma boa prática chamar isso no início para definir o título da aba e o ícone.
st.set_page_config(
    page_title="Análise de Colaborações com IA",
    page_icon="💡"
)

# --- FUNÇÕES DE LÓGICA ---

# Otimização: O Streamlit guarda o resultado desta função em cache.
# Se a função for chamada novamente com os mesmos argumentos, ele retorna o resultado
# salvo em vez de se reconectar ao banco, economizando tempo e recursos.
# O TTL (Time To Live) de 3600 segundos (1 hora) garante que os dados sejam atualizados a cada hora.
@st.cache_data(ttl=3600)
def fetch_data_from_db():
    """Conecta ao banco de dados usando os segredos do Streamlit e busca os dados."""
    try:
        # Conecta ao banco usando as credenciais salvas em st.secrets
        conn = psycopg2.connect(**st.secrets["postgres"])
        cursor = conn.cursor()
        
        # Busca dados dos últimos 90 dias
        query = "SELECT * FROM colaboracoes WHERE created_at >= NOW() - INTERVAL '90 days';"
        cursor.execute(query)
        
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        
        cursor.close()
        conn.close()
        
        # Converte para um DataFrame do Pandas, que é fácil de manipular
        df = pd.DataFrame(rows, columns=colnames)
        return df
        
    except psycopg2.OperationalError as e:
        st.error(f"Erro de Conexão com o Banco de Dados: {e}")
        st.info("Verifique se as credenciais no 'Secrets' do Streamlit estão corretas e se o IP do Streamlit Cloud tem permissão para acessar seu banco.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao buscar os dados: {e}")
        return None

def build_prompt(user_question, df):
    """Monta o prompt para o Gemini a partir da pergunta e dos dados."""
    
    # Converte o DataFrame para uma string em formato CSV, que é um ótimo formato para a IA ler.
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
        # Configura a API key a partir dos segredos do Streamlit
        genai.configure(api_key=st.secrets["google_api"]["key"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Erro ao chamar a API do Gemini: {e}")
        return None

# --- INTERFACE DO USUÁRIO (UI) ---

st.title("💡 Assistente de Análise de Colaborações")
st.markdown("Faça uma pergunta em linguagem natural sobre as colaborações dos últimos 90 dias e a IA irá gerar um insight para você.")

# Caixa de texto para a pergunta do usuário
default_question = "Qual cidade teve mais colaborações e qual o tipo de colaboração mais comum ('denuncia', 'sugestao', etc.)?"
user_question = st.text_area("Sua pergunta:", value=default_question, height=100)

# Botão para iniciar a análise
if st.button("Gerar Insight"):
    if not user_question:
        st.warning("Por favor, digite uma pergunta para análise.")
    else:
        # Mostra uma mensagem de "carregando" enquanto o processo acontece
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
