import streamlit as st

st.title("Classificador de Manifestações")

st.write("Insira o texto da manifestação para classificação:")

text_input = st.text_area("Manifestação", height=200)

if st.button("Classificar Manifestação"):
    if text_input:
        # Simula uma saída de classificação conforme o projeto.txt
        classification_result = {
          "especie_sugerida": "RECLAMAÇÃO",
          "natureza_sugerida": "ATENDIMENTO_RUIM",
          "confianca": "média",
          "justificativa": "Texto classificado como reclamação devido à insatisfação com a demora.",
          "elementos_identificados": ["demora_prazo", "insatisfação"]
        }
    
        st.subheader("Resultado da Classificação:")
        st.json(classification_result)
    else:
        st.warning("Por favor, insira um texto para classificar.")