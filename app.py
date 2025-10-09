import streamlit as st
import google.generativeai as genai
import json
 
# --- Funções Auxiliares ---
 
def get_gemini_classification(api_key, text):
    """
    Envia o texto para a API do Gemini e retorna a classificação em formato JSON.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
 
        # Prompt estruturado para guiar a IA a retornar o JSON no formato desejado
        prompt = f"""
        Analise a seguinte manifestação de ouvidoria e classifique-a estritamente no formato JSON abaixo.
        A resposta deve conter apenas o objeto JSON, sem nenhum texto ou formatação adicional como '```json'.

        Manifestação: "{text}"

        Formato de saída obrigatório:
        {{
          "especie_sugerida": "RECLAMAÇÃO, ELOGIO, SUGESTÃO, SOLICITAÇÃO ou DENÚNCIA",
          "natureza_sugerida": "Um resumo curto da natureza do problema em snake_case (ex: ATENDIMENTO_RUIM, DEMORA_PROCESSO)",
          "confianca": "alta, média ou baixa",
          "justificativa": "Uma frase curta explicando o porquê da classificação.",
          "elementos_identificados": ["lista", "de", "palavras-chave", "identificadas", "no", "texto"]
        }}
        """
 
        response = model.generate_content(prompt)
        
        # Limpa a resposta para garantir que seja um JSON válido
        cleaned_response = response.text.strip()
        
        return json.loads(cleaned_response)
 
    except json.JSONDecodeError:
        st.error(f"Erro: A IA retornou um formato inválido. Resposta recebida:\n\n{response.text}")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao chamar a API do Gemini: {str(e)}")
        return None
 
 
# --- Interface do Streamlit ---
 
st.set_page_config(page_title="Classificador de Manifestações", layout="wide")
 
# Barra lateral para configuração da API Key
with st.sidebar:
    st.header("Configuração")
    api_key = st.text_input("Sua API Key do Google Gemini", type="password", 
                            help="Obtenha sua chave em https://aistudio.google.com/app/apikey")
 
st.title("Classificador de Manifestações")
st.write("Esta ferramenta utiliza IA (Google Gemini) para analisar e sugerir uma classificação para manifestações de ouvidoria.")
 
text_input = st.text_area("Insira o texto da manifestação:", height=250, placeholder="Ex: 'Estou há dias esperando uma resposta sobre meu processo e ninguém me atende no telefone. Um absurdo a demora.'")
 
if st.button("Classificar Manifestação"):
    if not api_key:
        st.error("Erro: A API Key do Gemini não foi fornecida na barra lateral.")
    elif not text_input.strip():
        st.warning("Por favor, insira um texto para classificar.")
    else:
        with st.spinner("Analisando e classificando a manifestação..."):
            classification_result = get_gemini_classification(api_key, text_input)
            
            if classification_result:
                st.subheader("Resultado da Classificação (Sugestão da IA)")
                st.json(classification_result)