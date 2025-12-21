"""
Middleware de autenticación para proteger la API
"""
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from config.settings import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verifica que la API key proporcionada sea válida
    
    Args:
        api_key: API key del header
        
    Raises:
        HTTPException: Si la API key no es válida
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key faltante. Incluya el header 'X-API-Key'."
        )
    
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key inválida"
        )
    
    return api_key
