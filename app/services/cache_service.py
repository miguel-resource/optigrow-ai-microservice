"""
Módulo de manejo de caché para videos generados

Este módulo contiene:
- Almacenamiento temporal de videos en memoria
- Persistencia de caché en disco para debugging
- Carga de caché desde disco

Separado de gemini_service.py para mejor organización y mantenibilidad.
"""

import logging
import time
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Caché global de videos en memoria
_video_cache: Dict[str, Any] = {}


def get_video_cache() -> Dict[str, Any]:
    """
    Obtiene referencia al caché de videos
    
    Returns:
        Diccionario con el caché de videos
    """
    return _video_cache


def add_to_cache(key: str, video: Any) -> None:
    """
    Agrega un video al caché
    
    Args:
        key: Identificador único del video
        video: Objeto de video a cachear
    """
    _video_cache[key] = video
    logger.info(f"Video agregado al caché: {key}")


def get_from_cache(key: str) -> Optional[Any]:
    """
    Obtiene un video del caché
    
    Args:
        key: Identificador del video
        
    Returns:
        Objeto de video si existe, None si no
    """
    return _video_cache.get(key)


def remove_from_cache(key: str) -> bool:
    """
    Elimina un video del caché
    
    Args:
        key: Identificador del video
        
    Returns:
        True si se eliminó, False si no existía
    """
    if key in _video_cache:
        del _video_cache[key]
        logger.info(f"Video eliminado del caché: {key}")
        return True
    return False


def clear_cache() -> int:
    """
    Limpia todo el caché de videos
    
    Returns:
        Número de elementos eliminados
    """
    count = len(_video_cache)
    _video_cache.clear()
    logger.info(f"Caché limpiado: {count} videos eliminados")
    return count


def save_cache_to_disk(path: str = '/tmp/video_cache_debug.json') -> bool:
    """
    Guarda el caché actual en disco para debugging
    
    Args:
        path: Ruta donde guardar el archivo JSON
        
    Returns:
        True si se guardó exitosamente, False si falló
    """
    try:
        cache_data = {}
        for key, video in _video_cache.items():
            cache_data[key] = {
                "type": str(type(video)),
                "has_uri": hasattr(video, 'uri'),
                "uri": getattr(video, 'uri', None) if hasattr(video, 'uri') else None,
                "timestamp": time.time()
            }
        
        with open(path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Caché guardado en disco: {len(cache_data)} videos en {path}")
        return True
    except Exception as e:
        logger.error(f"Error guardando caché: {e}")
        return False


def load_cache_from_disk(path: str = '/tmp/video_cache_debug.json') -> Dict[str, Any]:
    """
    Intenta cargar caché desde disco para debugging
    
    Args:
        path: Ruta del archivo JSON a cargar
        
    Returns:
        Diccionario con datos del caché o vacío si falla
    """
    try:
        with open(path, 'r') as f:
            cache_data = json.load(f)
        
        logger.info(f"Caché cargado desde disco: {len(cache_data)} videos desde {path}")
        return cache_data
    except Exception as e:
        logger.warning(f"No se pudo cargar caché desde disco: {e}")
        return {}


def get_cache_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas del caché
    
    Returns:
        Diccionario con estadísticas del caché
    """
    stats = {
        "total_videos": len(_video_cache),
        "videos_with_uri": sum(1 for v in _video_cache.values() if hasattr(v, 'uri')),
        "videos_with_gcs_uri": sum(1 for v in _video_cache.values() if hasattr(v, 'gcs_uri')),
        "keys": list(_video_cache.keys())
    }
    return stats


# Clases auxiliares para representar videos en caché
class ConcatenatedVideo:
    """Representa un video concatenado almacenado en GCS"""
    def __init__(self, uri: str):
        self.uri = uri
        self.gcs_uri = uri


class LocalConcatenatedVideo:
    """Representa un video concatenado almacenado localmente"""
    def __init__(self, local_path: str):
        self.uri = f"file://{local_path}"
        self.local_path = local_path
