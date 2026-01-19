"""
Módulo de servicios de almacenamiento en Google Cloud Storage

Este módulo contiene:
- Descarga de videos desde GCS
- Generación de URLs firmadas temporales
- Generación de URIs de salida para GCS
- Subida de videos a GCS

Separado de gemini_service.py para mejor organización y mantenibilidad.
"""

import logging
import os
import uuid
import time
import datetime
from typing import Optional

from google.cloud import storage
from config.settings import settings

logger = logging.getLogger(__name__)


def get_storage_client() -> storage.Client:
    """
    Crea y devuelve un cliente de Google Cloud Storage
    
    Returns:
        Cliente de GCS configurado
    """
    if settings.gemini_project_id:
        return storage.Client(project=settings.gemini_project_id)
    return storage.Client()


def parse_gcs_uri(gcs_uri: str) -> tuple:
    """
    Parsea una URI de GCS a bucket y blob name
    
    Args:
        gcs_uri: URI en formato gs://bucket/path/file.mp4
        
    Returns:
        Tupla (bucket_name, blob_name)
        
    Raises:
        ValueError: Si la URI no es válida
    """
    if not gcs_uri or not gcs_uri.startswith('gs://'):
        raise ValueError(f"URI inválida, debe empezar con 'gs://': {gcs_uri}")
    
    path_parts = gcs_uri.replace("gs://", "").split("/", 1)
    if len(path_parts) != 2:
        raise ValueError(f"Formato de URI GCS inválido: {gcs_uri}")
    
    return path_parts[0], path_parts[1]


def download_video_from_gcs(gcs_uri: str, output_path: str) -> bool:
    """
    Descarga un video directamente desde GCS usando las credenciales configuradas
    
    Args:
        gcs_uri: URI del video en formato gs://bucket/path/video.mp4
        output_path: Ruta local donde guardar el video
        
    Returns:
        True si la descarga fue exitosa, False si falló
    """
    try:
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        
        logger.info(f"Descargando desde GCS: bucket={bucket_name}, blob={blob_name}")
        
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Verificar que el blob existe
        if not blob.exists():
            logger.error(f"El video no existe en GCS: {gcs_uri}")
            return False
        
        # Descargar el video
        blob.download_to_filename(output_path)
        
        # Verificar que se descargó correctamente
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Video descargado exitosamente: {os.path.getsize(output_path)} bytes")
            return True
        else:
            logger.error("La descarga de GCS resultó en un archivo vacío o inexistente")
            return False
            
    except ValueError as e:
        logger.error(f"URI GCS inválida: {e}")
        return False
    except Exception as e:
        logger.error(f"Error descargando desde GCS: {e}")
        return False


def generate_signed_url(gcs_uri: str, expiration_minutes: int = 15) -> Optional[str]:
    """
    Genera una URL firmada temporal para un video almacenado en Google Cloud Storage
    
    Args:
        gcs_uri: URI del formato gs://bucket/path/video.mp4
        expiration_minutes: Tiempo de expiración en minutos (default: 15)
        
    Returns:
        URL firmada temporal para acceder al video, None si falla
    """
    try:
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        
        logger.info(f"Generando URL firmada para: bucket={bucket_name}, blob={blob_name}")
        
        try:
            storage_client = get_storage_client()
            if settings.gemini_project_id:
                logger.info(f"Usando proyecto explícito: {settings.gemini_project_id}")
            else:
                logger.warning("No se configuró GEMINI_PROJECT_ID, usando proyecto por defecto")
        except Exception as client_error:
            logger.error(f"Error creando cliente GCS: {client_error}")
            return None
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Verificar que el blob existe
        if not blob.exists():
            logger.warning(f"El video no existe en GCS: {gcs_uri}")
            return None
        
        # Generar URL firmada válida por el tiempo especificado
        try:
            url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(minutes=expiration_minutes),
                method="GET",
            )
            
            logger.info(f"URL firmada generada exitosamente (expira en {expiration_minutes}min)")
            return url
            
        except Exception as signing_error:
            error_msg = str(signing_error).lower()
            
            # Error específico de credenciales
            if "private key" in error_msg or "service account" in error_msg:
                logger.warning(f"No se pueden firmar URLs con credenciales actuales: {signing_error}")
                logger.info("Para URLs firmadas, configura Service Account. Ver SERVICE_ACCOUNT_SETUP.md")
                logger.info("Alternativa temporal: Hacer bucket público con 'gsutil iam ch allUsers:objectViewer gs://ai_microservice_videos'")
                
                # Devolver URL pública si el objeto es accesible públicamente
                try:
                    public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
                    logger.info(f"Intentando URL pública: {public_url}")
                    return public_url
                except:
                    return None
            else:
                logger.error(f"Error firmando URL: {signing_error}")
                return None
        
    except ValueError as e:
        logger.error(f"Error en URI: {e}")
        return None
    except Exception as e:
        logger.error(f"Error generando URL firmada para {gcs_uri}: {str(e)}")
        return None


def generate_gcs_output_uri(bucket_name: str, base_path: str = "videos/generated/") -> str:
    """
    Genera una URI de salida para almacenar videos en Google Cloud Storage
    
    Args:
        bucket_name: Nombre del bucket (sin gs://)
        base_path: Ruta base dentro del bucket
        
    Returns:
        URI completa de GCS en formato gs://bucket/path/video_timestamp.mp4
        
    Raises:
        ValueError: Si bucket_name está vacío
    """
    if not bucket_name:
        raise ValueError("bucket_name no puede estar vacío")
    
    # Generar nombre único del archivo
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    filename = f"video_{timestamp}_{unique_id}.mp4"
    
    # Asegurar que base_path termina con /
    if base_path and not base_path.endswith('/'):
        base_path += '/'
    
    # Construir URI completa
    gcs_uri = f"gs://{bucket_name}/{base_path}{filename}"
    
    logger.info(f"URI de salida GCS generada: {gcs_uri}")
    return gcs_uri


def upload_video_to_gcs(
    local_path: str, 
    bucket_name: str, 
    blob_path: str,
    content_type: str = 'video/mp4'
) -> Optional[str]:
    """
    Sube un video local a Google Cloud Storage
    
    Args:
        local_path: Ruta local del video
        bucket_name: Nombre del bucket GCS
        blob_path: Ruta del blob en el bucket
        content_type: Tipo MIME del archivo
        
    Returns:
        URI del video subido o None si falla
    """
    try:
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        with open(local_path, 'rb') as video_file:
            blob.upload_from_file(video_file, content_type=content_type)
        
        gcs_uri = f"gs://{bucket_name}/{blob_path}"
        logger.info(f"Video subido exitosamente a GCS: {gcs_uri}")
        return gcs_uri
        
    except Exception as e:
        logger.error(f"Error subiendo video a GCS: {e}")
        return None


def gcs_uri_to_https_url(gcs_uri: str) -> str:
    """
    Convierte una URI gs:// a una URL HTTPS pública
    
    Args:
        gcs_uri: URI en formato gs://bucket/path/file.mp4
        
    Returns:
        URL HTTPS pública
    """
    if not gcs_uri.startswith('gs://'):
        return gcs_uri
    
    bucket_path = gcs_uri.replace('gs://', '')
    return f"https://storage.googleapis.com/{bucket_path}"


def check_blob_exists(gcs_uri: str) -> bool:
    """
    Verifica si un blob existe en GCS
    
    Args:
        gcs_uri: URI del blob
        
    Returns:
        True si existe, False si no
    """
    try:
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        logger.error(f"Error verificando existencia de blob: {e}")
        return False
