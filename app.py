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
    """Conecta-se à base de dados MySQL usando os segredos e busca todos os dados."""
    try:
        conn = mysql.connector.connect(**st.secrets["mysql"])
        query = "SELECT * FROM colaboracoes;"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Ocorreu um erro ao conectar-se ou buscar dados: {e}")
        return None

def generate_insight_huggingface(user_question, df):
    """
    Envia o prompt para a API do Hugging Face usando o método de conversação correto
    e instruções refinadas para evitar alucinações.
    """
    # ALTERAÇÃO: Prioriza os modelos mais poderosos que o utilizador já autorizou.
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

    data_csv = df.to_csv(index=False)

    # ALTERAÇÃO: Instruções do sistema muito mais rigorosas e detalhadas.
    system_prompt = """Você é um analista de dados de elite. A sua única função é analisar os dados em formato CSV que o utilizador fornece e responder à pergunta dele.
    Siga estes passos rigorosamente:
    1.  Leia atentamente a pergunta do utilizador para entender o objetivo.
    2.  Examine os dados CSV fornecidos para encontrar as informações relevantes.
    3.  Formule uma resposta clara, profissional e concisa, em português.
    A sua resposta deve ser baseada **exclusivamente** nos dados. Não invente informações. Não gere código. Não inclua a sua reflexão sobre os passos, apenas a resposta final."""

    user_prompt = f"""
Aqui estão os dados para análise:
--- DADOS CSV ---
{data_csv}

--- PERGUNTA ---
Com base **exclusivamente** nos dados acima, responda à seguinte pergunta: {user_question}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for model_id in candidate_models:
        st.info(f"A testar o modelo de IA: {model_id}...")
        
        try:
            client = InferenceClient(model=model_id, token=api_token)
            # ALTERAÇÃO: Adicionado 'temperature' para reduzir a aleatoriedade e 'top_p' para focar em respostas prováveis.
            response = client.chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.5, 
                top_p=0.95
            )
            
            insight = response.choices[0].message.content
            
            st.success(f"Modelo '{model_id}' respondeu com sucesso!")
            return insight.strip()

        except Exception as e:
            error_message = str(e)
            if "401" in error_message:
                 st.error(f"Erro de Autenticação (401) com o modelo '{model_id}'. O seu token de API é inválido. Por favor, verifique-o nos 'Secrets'.")
                 return None
            else:
                 st.warning(f"O modelo '{model_id}' falhou com um erro: {error_message}. A tentar o próximo modelo...")
            continue
    
    st.error("Não foi possível obter uma resposta de nenhum dos modelos de IA disponíveis. Por favor, tente novamente mais tarde.")
    return None

# --- INTERFACE DO UTILIZADOR (UI) ---

st.title("💡 Assistente de Análise de Colaborações")
st.markdown("Faça uma pergunta sobre o **histórico completo** de colaborações e a IA irá gerar um insight para si.")
st.info("ℹ️ Esta demonstração usa modelos da comunidade Hugging Face. A primeira geração pode demorar mais enquanto o modelo é carregado.")

default_question = "Qual cidade teve mais colaborações e qual o tipo de colaboração mais comum ('denuncia', 'sugestao', etc.)?"
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
                st.warning(f"A analisar o histórico completo de {len(dados_df)} colaborações. A geração da resposta pode demorar mais tempo.")
                
                with st.spinner("A contactar os modelos de IA do Hugging Face..."):
                    # ALTERAÇÃO: Removida a função 'build_user_prompt' e a chamada foi simplificada.
                    insight = generate_insight_huggingface(user_question, dados_df)

                if insight:
                    st.subheader("Análise Gerada pela IA:")
                    st.markdown(insight)

