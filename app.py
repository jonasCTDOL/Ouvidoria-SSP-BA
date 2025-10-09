import streamlit as st
import json

st.title("Classificador de Manifestações")

st.write("Enter text for classification:")

text_input = st.text_area("Manifestação", height=200)

if st.button("Classificar"):
    # Simulate a classification output
    classification_result = {
        "category": "Simulated Category",
        "score": 0.85,
        "explanation": "This is a simulated explanation of the classification."
    }

    st.subheader("Classification Results:")
    st.json(classification_result)