import streamlit as st

def customMsg(msg, type_='warning'):
    
    placeholder = st.empty()
    styledMsg = f"""<div class="element-container" style="width: 693px;">
                    <div class="alert alert-{type_} stAlert" style="width: 693px;">
                    <div class="markdown-text-container">
                    <p>{msg}</p></div></div></div>"""
                    
    placeholder.markdown(styledMsg, unsafe_allow_html=True)
    
    return placeholder