"""
Configuración del microservicio
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "OptiGrow AI Microservice"
    api_version: str = "1.0.0"
    
    # Security
    api_key: str = "default-api-key-change-in-production"
    
    # Gemini Configuration
    gemini_api_key: str = ""
    
    # CORS Settings
    allowed_origins: str = "http://localhost:8000"
    
    # Environment
    environment: str = "development"
    
    @property
    def origins_list(self) -> List[str]:
        """Convierte la cadena de orígenes permitidos en una lista"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Instancia global de configuración
settings = Settings()
