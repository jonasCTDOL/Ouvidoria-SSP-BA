import streamlit as st
import google.generativeai as genai
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Funções Auxiliares ---

def get_gemini_classification(api_key, text):
    """
    Envia o texto para a API do Gemini e retorna a classificação em formato JSON.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.0-pro')

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

def send_email(sender_email, sender_password, recipient_email, subject, body):
    """
    Envia um e-mail com o conteúdo da manifestação e a análise da IA.
    """
    try:
        # Configuração do servidor SMTP do Hotmail/Outlook
        smtp_server = "smtp.office365.com"
        smtp_port = 587

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        st.success(f"E-mail enviado com sucesso para {recipient_email}!")

    except Exception as e:
        st.error(f"Ocorreu um erro ao enviar o e-mail: {str(e)}")

# --- Interface do Streamlit ---

st.set_page_config(page_title="Classificador de Manifestações", layout="wide")

# Barra lateral para configuração
with st.sidebar:
    st.header("Configuração")
    api_key = st.text_input("Sua API Key do Google Gemini", type="password", 
                            help="Obtenha sua chave em https://aistudio.google.com/app/apikey")
    st.header("Configuração de E-mail")
    sender_email = st.text_input("Seu E-mail (Remetente)", help="Ex: seu_email@hotmail.com")
    sender_password = st.text_input("Sua Senha de Aplicativo", type="password", 
                                    help="Use uma senha de aplicativo se tiver autenticação de dois fatores.")

st.title("Classificador de Manifestações e Notificação por E-mail")
st.write("Esta ferramenta utiliza IA (Google Gemini) para analisar, classificar manifestações de ouvidoria e enviar um relatório por e-mail.")

text_input = st.text_area("Insira o texto da manifestação:", height=250, placeholder="Ex: 'Estou há dias esperando uma resposta sobre meu processo e ninguém me atende no telefone. Um absurdo a demora.'")

if st.button("Classificar e Enviar E-mail"):
    # Validações
    if not api_key:
        st.error("Erro: A API Key do Gemini não foi fornecida na barra lateral.")
    elif not sender_email or not sender_password:
        st.error("Erro: As configurações de e-mail (remetente e senha) não foram fornecidas.")
    elif not text_input.strip():
        st.warning("Por favor, insira um texto para classificar.")
    else:
        with st.spinner("Analisando, classificando e enviando e-mail..."):
            classification_result = get_gemini_classification(api_key, text_input)
            
            if classification_result:
                st.subheader("Resultado da Classificação (Sugestão da IA)")
                st.json(classification_result)

                # Prepara e envia o e-mail
                recipient_email = "mouzartti@hotmail.com"
                subject = "Análise de Manifestação da Ouvidoria"
                body = f"""
                Prezados,

                Segue a análise de uma nova manifestação recebida.

                ---
                Texto da Manifestação Original:
                ---
                {text_input}

                ---
                Análise da IA (Gemini):
                ---
                {json.dumps(classification_result, indent=2, ensure_ascii=False)}

                Atenciosamente,
                Sistema de Análise de Ouvidoria
                """
                send_email(sender_email, sender_password, recipient_email, subject, body)
