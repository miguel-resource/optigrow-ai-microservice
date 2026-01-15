"""
Configuración del microservicio
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    api_title: str = "OptiGrow AI Microservice"
    api_version: str = "1.0.0"
    
    # Security
    api_key: str = "default-api-key-change-in-production"
    
    # Gemini Configuration
    gemini_api_key: str = ""
    gemini_project_id: str = ""
    gemini_location: str = "us-central1"
    gemini_vertexai: bool = False
    
    # Google Cloud Storage Configuration (para videos largos)
    gcs_bucket_name: str = ""  # Nombre del bucket sin gs://, ej: "mi-bucket-videos"
    gcs_output_path: str = "videos/generated/"  # Ruta dentro del bucket
    
    # CORS Settings
    allowed_origins: str = "http://localhost:8000"
    
    # Environment
    environment: str = "development"
    
    @property
    def origins_list(self) -> List[str]:
        """Convierte la cadena de orígenes permitidos en una lista"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False
    )


# Instancia global de configuración
settings = Settings()
