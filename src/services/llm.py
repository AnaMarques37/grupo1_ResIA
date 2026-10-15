import os
import logging
from typing import Any
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def obter_credenciais() -> tuple[str, str, str, int]:
    """Retorna dinamicamente (base_url, api_key, model_name, timeout)."""
    # 1. Tenta obter a chave Groq do ambiente ou do Streamlit Secrets
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                groq_key = str(st.secrets["GROQ_API_KEY"]).strip()
        except Exception:
            pass

    if groq_key:
        # Nuvem: Groq
        base_url = "https://api.groq.com/openai/v1"
        api_key = groq_key
        # Modelo padrão na Groq com suporte excelente a JSON Schema e português
        model_name = os.getenv("LLM_MODEL", "").strip() or "llama-3.3-70b-versatile"
        timeout = 60
        logger.info(f"LLM: Usando API Groq (modelo: {model_name})")
        return base_url, api_key, model_name, timeout
    else:
        # Local: LM Studio / Bionic
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or "http://localhost:1234/v1"
        api_key = "lm-studio"
        model_name = os.getenv("LLM_MODEL", "").strip() or "qwen2.5-7b"
        timeout = int(os.getenv("LLM_TIMEOUT", "120"))
        logger.info(f"LLM: Usando Servidor Local em {base_url} (modelo: {model_name})")
        return base_url, api_key, model_name, timeout


class LLMProxy:
    """Proxy que instancia o ChatOpenAI dinamicamente para garantir a leitura correta dos segredos."""

    def __init__(self, temperatura: float):
        self.temperatura = temperatura
        self._cliente_instanciado = None

    def _obter_cliente(self) -> ChatOpenAI:
        base_url, api_key, model_name, timeout = obter_credenciais()
        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            temperature=self.temperatura,
            timeout=timeout,
            max_retries=1,
        )

    def invoke(self, *args, **kwargs) -> Any:
        return self._obter_cliente().invoke(*args, **kwargs)

    def bind(self, *args, **kwargs) -> Any:
        return self._obter_cliente().bind(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._obter_cliente(), name)


llm_sintetizador = LLMProxy(0.0)   # Sintetizador
llm_socratico = LLMProxy(0.7)      # Agente Socrático
llm_texto = LLMProxy(0.0)          # Agente de Texto
