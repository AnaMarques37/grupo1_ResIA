import os

from langchain_openai import ChatOpenAI

try:
    import streamlit as st
    STREAMLIT_SECRETS = getattr(st, "secrets", {})
except Exception:
    STREAMLIT_SECRETS = {}


def obter_chave_groq() -> str:
    """Busca a chave da Groq nas variáveis de ambiente ou nos segredos do Streamlit."""
    return os.getenv("GROQ_API_KEY", "").strip() or STREAMLIT_SECRETS.get("GROQ_API_KEY", "")


GROQ_API_KEY = obter_chave_groq()

if GROQ_API_KEY:
    # Configuração em Nuvem (Groq - Ultra rápido e gratuito)
    BASE_URL = "https://api.groq.com/openai/v1"
    API_KEY = GROQ_API_KEY
    LOCAL_MODEL = os.getenv("LLM_MODEL", "").strip() or "llama-3.3-70b-versatile"
    TIMEOUT_S = 60
else:
    # Configuração Local (LM Studio / Bionic)
    def modelo_local() -> str:
        return os.getenv("LLM_MODEL", "").strip() or "qwen2.5-7b"

    LOCAL_MODEL = modelo_local()
    BASE_URL = "http://localhost:1234/v1"
    API_KEY = "lm-studio"
    TIMEOUT_S = 120


def _cliente(temperatura: float) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=LOCAL_MODEL,
        temperature=temperatura,
        timeout=TIMEOUT_S,
        max_retries=0,
    )


llm_sintetizador = _cliente(0)  # Sintetizador
llm_socratico = _cliente(0.7)   # Agente Socrático
llm_texto = _cliente(0)         # Agente de Texto
