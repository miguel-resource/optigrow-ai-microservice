"""
Tests para el servicio de Gemini
"""
import pytest
from config.settings import settings
from app.services.gemini_service import GeminiService


@pytest.mark.asyncio
async def test_gemini_service_initialization():
    """Test de inicialización del servicio"""
    service = GeminiService(api_key=settings.gemini_api_key, model_name="gemini-2.5-flash-lite")
    assert service.api_key == settings.gemini_api_key
    assert service.model_name == "gemini-2.5-flash-lite"
    assert service.name == "gemini"


@pytest.mark.asyncio
async def test_gemini_service_name():
    """Test de la propiedad name"""
    service = GeminiService(api_key=settings.gemini_api_key)
    assert service.name == "gemini"