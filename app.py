import streamlit as st
import mysql.connector
import pandas as pd
from huggingface_hub import InferenceClient
import re

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
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at']).dt.date
        return df
    except Exception as e:
        st.error(f"Ocorreu um erro ao conectar-se ou buscar dados: {e}")
        return None

def get_data_summary(df):
    """Cria um resumo estatístico e informativo do DataFrame para perguntas gerais."""
    summary_parts = []
    summary_parts.append(f"Resumo Geral: {len(df)} colaborações no total, de {df['created_at'].min()} a {df['created_at'].max()}.\n")
    
    for col in ['tipo_colaboracao', 'cidade', 'estado', 'status']:
        if col in df.columns:
            summary_parts.append(f"Contagem por '{col}':\n{df[col].value_counts().to_string()}\n")
            
    return "\n".join(summary_parts)

def filter_relevant_data(df, question):
    """
    Filtra o DataFrame para encontrar apenas os registos relevantes para a pergunta do utilizador.
    Se a pergunta for geral, retorna None para que um resumo seja usado.
    """
    general_keywords = ['quantos', 'qual', 'resumo', 'geral', 'total', 'lista', 'liste', 'quais']
    if any(keyword in question.lower() for keyword in general_keywords):
        return None

    keywords = set(re.findall(r'\b\w{3,}\b', question.lower()))
    
    search_cols = ['descricao', 'observacoes', 'cidade', 'bairro', 'rua', 'tipo_colaboracao']
    
    mask = pd.Series([False] * len(df))
    for col in search_cols:
        if col in df.columns:
            str_col = df[col].astype(str).str.lower()
            for keyword in keywords:
                mask |= str_col.str.contains(keyword, na=False)
    
    relevant_df = df[mask]
    return relevant_df if not relevant_df.empty else None

def generate_insight_huggingface(user_question, df):
    """
    Envia um contexto otimizado (dados filtrados ou resumo) para a API para obter a melhor resposta.
    """
    candidate_models = ["meta-llama/Meta-Llama-3-8B-Instruct", "google/gemma-2-9b-it", "HuggingFaceH4/zephyr-7b-beta"]
    
    try:
        api_token = st.secrets["huggingface_api"]["token"]
        st.info(f"A usar o token que começa com '{api_token[:6]}' e termina com '{api_token[-4:]}'.")
    except Exception as e:
        st.error("Erro ao ler o token da API. Verifique os seus 'Secrets'.")
        return None

    relevant_data = filter_relevant_data(df, user_question)
    
    if relevant_data is not None:
        st.info(f"Filtro Inteligente: {len(relevant_data)} registos relevantes encontrados para a sua pergunta.")
        context_data = relevant_data.to_csv(index=False)
        # ALTERAÇÃO: Instruções refinadas para um contexto policial.
        system_prompt = """Você é um assistente de inteligência policial. A sua missão é analisar denúncias e colaborações de cidadãos para identificar potenciais atividades criminosas e fornecer resumos objetivos para agentes da lei.
Analise os DADOS FILTRADOS em formato CSV abaixo, que são denúncias relevantes para a pergunta do agente.
Formule uma resposta detalhada, objetiva e factual, focada em extrair informações úteis para uma investigação. A sua análise é confidencial e para uso exclusivo da polícia."""
        user_prompt_content = f"Com base **exclusivamente** nos dados filtrados abaixo, responda à pergunta do agente: {user_question}\n\n--- DADOS FILTRADOS ---\n{context_data}"
    else:
        st.info("A sua pergunta é geral. A IA irá usar um resumo estatístico para responder.")
        context_data = get_data_summary(df)
        # ALTERAÇÃO: Instruções refinadas para um contexto policial.
        system_prompt = """Você é um assistente de inteligência policial. A sua missão é analisar dados para identificar tendências e padrões em colaborações de cidadãos.
Analise o RESUMO ESTATÍSTICO abaixo e use-o para responder à pergunta do agente de forma clara e profissional, focada em insights de segurança pública."""
        user_prompt_content = f"Com base **exclusivamente** no resumo abaixo, responda à pergunta do agente: {user_question}\n\n--- RESUMO ESTATÍSTICO ---\n{context_data}"

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_content}]

    for model_id in candidate_models:
        st.info(f"A testar o modelo de IA: {model_id}...")
        try:
            client = InferenceClient(model=model_id, token=api_token)
            response = client.chat_completion(messages=messages, max_tokens=1024, temperature=0.3, top_p=0.95)
            insight = response.choices[0].message.content
            st.success(f"Modelo '{model_id}' respondeu com sucesso!")
            return insight.strip()
        except Exception as e:
            st.warning(f"O modelo '{model_id}' falhou com um erro: {e}. A tentar o próximo modelo...")
    
    st.error("Não foi possível obter uma resposta de nenhum dos modelos de IA disponíveis.")
    return None

# --- INTERFACE DO UTILIZADOR (UI) ---

st.title("💡 Assistente de Análise de Colaborações")
st.markdown("Faça uma pergunta sobre o **histórico completo** de colaborações e a IA irá gerar um insight para si.")
st.info("ℹ️ A aplicação possui um **Filtro Inteligente** que analisa a sua pergunta para fornecer respostas mais precisas.")

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
                with st.spinner("O Filtro Inteligente está a processar a sua pergunta и a contactar a IA..."):
                    insight = generate_insight_huggingface(user_question, dados_df)
                if insight:
                    st.subheader("Análise Gerada pela IA:")
                    st.markdown(insight)

