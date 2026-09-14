#%% Pakete
import streamlit as st

st.title("Filmvorhersage")

#%% ChatInput widget
prompt = st.chat_input("Beschreibe die Filmhandlung!")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")
