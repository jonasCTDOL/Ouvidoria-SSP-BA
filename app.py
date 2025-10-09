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

def generate_insight_huggingface(user_question, df):
    """
    Envia os dados completos para a API do Hugging Face para obter uma resposta inteligente e profunda.
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

    # ALTERAÇÃO: Voltamos a usar o CSV completo para uma análise profunda.
    data_csv = df.to_csv(index=False)

    # ALTERAÇÃO: Instruções do sistema muito mais rigorosas e detalhadas.
    system_prompt = """Você é um analista de dados de elite. A sua única função é analisar os dados em formato CSV que o utilizador fornece e responder à pergunta dele de forma clara, profissional e detalhada, em português.
Siga estes passos rigorosamente:
1.  Leia atentamente a pergunta do utilizador para entender o objetivo da análise.
2.  Examine **todas as colunas** dos dados CSV fornecidos para encontrar as informações relevantes, prestando especial atenção a colunas de texto livre como 'descricao' e 'observacoes'.
3.  Formule uma resposta completa e baseada em factos. Se a pergunta for sobre um tema específico, procure por palavras-chave relevantes nos dados.
A sua resposta deve ser baseada **exclusivamente** nos dados. Não invente informações. Não gere código."""

    user_prompt = f"""
Aqui estão os dados completos para análise:
--- DADOS CSV ---
{data_csv}

--- PERGUNTA DO UTILIZADOR ---
Com base **exclusivamente** em todos os dados acima, responda à seguinte pergunta: {user_question}
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
                temperature=0.3, # Mantém a temperatura baixa para respostas factuais
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
# ALTERAÇÃO: Mensagem de informação atualizada.
st.info("ℹ️ A aplicação envia agora todos os dados para a IA para permitir respostas mais profundas e precisas.")

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
                
                # ALTERAÇÃO: Mensagem do spinner atualizada.
                with st.spinner("A enviar os dados completos para a IA e a aguardar a análise..."):
                    insight = generate_insight_huggingface(user_question, dados_df)

                if insight:
                    st.subheader("Análise Gerada pela IA:")
                    st.markdown(insight)

