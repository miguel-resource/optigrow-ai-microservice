"""
Módulo de procesamiento de videos con FFmpeg

Este módulo contiene:
- Concatenación de videos con FFmpeg
- Verificación de FFmpeg
- Descarga y procesamiento de segmentos
- Manejo de archivos temporales

Separado de gemini_service.py para mejor organización y mantenibilidad.
"""

import logging
import os
import subprocess
import tempfile
import shutil
import time
import uuid
from typing import Dict, Any, List, Optional

import requests

from config.settings import settings
from app.services.storage_service import (
    download_video_from_gcs,
    generate_signed_url,
    gcs_uri_to_https_url,
    get_storage_client
)
from app.services.cache_service import (
    add_to_cache,
    ConcatenatedVideo,
    LocalConcatenatedVideo
)

logger = logging.getLogger(__name__)


def check_ffmpeg_available() -> bool:
    """
    Verifica si FFmpeg está disponible en el sistema
    
    Returns:
        True si FFmpeg está disponible, False si no
        
    Raises:
        Exception: Si FFmpeg no está instalado o no responde
    """
    try:
        ffmpeg_check = subprocess.run(
            ['ffmpeg', '-version'], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if ffmpeg_check.returncode != 0:
            raise Exception("FFmpeg no está instalado o no está disponible en el PATH")
        logger.info("FFmpeg detectado correctamente")
        return True
    except FileNotFoundError:
        raise Exception("FFmpeg no está instalado. Instálalo con: apt-get install ffmpeg (Linux) o brew install ffmpeg (macOS)")
    except subprocess.TimeoutExpired:
        raise Exception("FFmpeg no responde, verificar instalación")


def download_video_segment(
    video_url: str, 
    output_path: str, 
    video_result: Dict[str, Any],
    max_retries: int = 3
) -> bool:
    """
    Descarga un segmento de video con reintentos y fallback a GCS directo
    
    Args:
        video_url: URL HTTP del video
        output_path: Ruta local donde guardar el video
        video_result: Resultado de la generación con metadata
        max_retries: Número máximo de reintentos
        
    Returns:
        True si la descarga fue exitosa
        
    Raises:
        Exception: Si la descarga falla después de todos los reintentos
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(video_url, timeout=300)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Segmento descargado exitosamente ({len(response.content)} bytes)")
            return True
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(f"Error descargando segmento (intento {attempt + 1}): {e}")
                
                # Si es error 403, intentar descarga directa desde GCS
                if ("403" in str(e) or "Forbidden" in str(e)) and 'video_uri' in video_result:
                    logger.warning("Error 403 - intentando descarga directa desde GCS...")
                    gcs_uri = video_result.get('video_uri', '')
                    if gcs_uri.startswith('gs://'):
                        if download_video_from_gcs(gcs_uri, output_path):
                            logger.info("Segmento descargado exitosamente desde GCS directo")
                            return True
                
                time.sleep(2 ** attempt)  # Backoff exponencial
            else:
                # Último intento: intentar descarga directa como fallback final
                if 'video_uri' in video_result:
                    gcs_uri = video_result.get('video_uri', '')
                    if gcs_uri.startswith('gs://'):
                        logger.warning("Último intento: descarga directa desde GCS...")
                        if download_video_from_gcs(gcs_uri, output_path):
                            logger.info("Segmento descargado en último intento desde GCS")
                            return True
                
                raise Exception(f"Fallo descarga después de {max_retries} intentos: {e}")
    
    return False


def concatenate_videos_ffmpeg(
    video_paths: List[str],
    output_path: str,
    timeout: int = 300
) -> bool:
    """
    Concatena múltiples videos usando FFmpeg
    
    Args:
        video_paths: Lista de rutas a los videos a concatenar
        output_path: Ruta del video de salida
        timeout: Timeout en segundos para el proceso FFmpeg
        
    Returns:
        True si la concatenación fue exitosa
        
    Raises:
        Exception: Si FFmpeg falla
    """
    temp_dir = os.path.dirname(output_path)
    filelist_path = os.path.join(temp_dir, 'filelist.txt')
    
    # Crear lista de archivos para FFmpeg
    with open(filelist_path, 'w') as f:
        for video_path in video_paths:
            f.write(f"file '{video_path}'\n")
    
    # Concatenar con FFmpeg (comando optimizado)
    logger.info("Concatenando videos con FFmpeg...")
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Sobrescribir archivo de salida
        '-f', 'concat', 
        '-safe', '0', 
        '-i', filelist_path,
        '-c', 'copy',  # Copiar streams sin re-encodificar
        '-avoid_negative_ts', 'make_zero',  # Evitar problemas de timestamp
        output_path
    ]
    
    logger.info(f"Comando FFmpeg: {' '.join(ffmpeg_cmd)}")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=timeout)
    
    if result.returncode != 0:
        logger.error(f"FFmpeg stdout: {result.stdout}")
        logger.error(f"FFmpeg stderr: {result.stderr}")
        raise Exception(f"Error en FFmpeg (código {result.returncode}): {result.stderr}")
    
    logger.info("Concatenación FFmpeg completada exitosamente")
    
    # Verificar que el archivo de salida existe y tiene contenido
    if not os.path.exists(output_path):
        raise Exception("FFmpeg no generó el archivo de salida")
    
    output_size = os.path.getsize(output_path)
    if output_size == 0:
        raise Exception("FFmpeg generó un archivo vacío")
        
    logger.info(f"Video concatenado generado: {output_size} bytes")
    return True


def get_video_url_from_result(result: Dict[str, Any]) -> str:
    """
    Extrae la URL de video de un resultado de generación
    
    Args:
        result: Resultado de generate_video_from_images
        
    Returns:
        URL HTTP del video
    """
    video_url = result.get('signed_url', '')
    
    if not video_url:
        video_uri = result.get('video_uri', '')
        if video_uri and video_uri.startswith('gs://'):
            video_url = gcs_uri_to_https_url(video_uri)
        else:
            video_url = video_uri
    
    return video_url


def upload_concatenated_to_gcs(
    local_path: str,
    bucket_name: str,
    output_path: str
) -> Optional[str]:
    """
    Sube un video concatenado a GCS
    
    Args:
        local_path: Ruta local del video
        bucket_name: Nombre del bucket GCS
        output_path: Ruta base en el bucket
        
    Returns:
        URI del video subido o None si falla
    """
    try:
        storage_client = get_storage_client()
        if settings.gemini_project_id:
            logger.info(f"Usando proyecto para upload: {settings.gemini_project_id}")
        else:
            logger.warning("No hay project ID configurado - usando credenciales por defecto")
        
        bucket = storage_client.bucket(bucket_name)
        
        final_filename = f"video_concatenated_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
        blob_path = f"{output_path}/{final_filename}"
        blob = bucket.blob(blob_path)
        
        with open(local_path, 'rb') as video_file:
            blob.upload_from_file(video_file, content_type='video/mp4')
        
        final_uri = f"gs://{bucket_name}/{blob_path}"
        logger.info(f"Video concatenado subido exitosamente: {final_uri}")
        return final_uri
        
    except Exception as e:
        logger.error(f"Error subiendo video concatenado a GCS: {e}")
        return None


def process_concatenated_video(
    segment_results: List[Dict[str, Any]],
    segment_duration: int,
    fps: int,
    aspect_ratio: str,
    transition_style: str,
    motion_strength: float,
    total_images: int
) -> Dict[str, Any]:
    """
    Procesa y concatena múltiples segmentos de video
    
    Args:
        segment_results: Lista de resultados de generate_video_from_images
        segment_duration: Duración de cada segmento en segundos
        fps: Frames por segundo
        aspect_ratio: Relación de aspecto
        transition_style: Estilo de transición
        motion_strength: Fuerza del movimiento
        total_images: Número total de imágenes usadas
        
    Returns:
        Diccionario con información del video concatenado
    """
    # Verificar FFmpeg
    check_ffmpeg_available()
    
    # Obtener URLs de videos
    video_urls = [get_video_url_from_result(r) for r in segment_results]
    
    if not all(video_urls):
        raise Exception("No se pudieron obtener las URLs HTTP de todos los segmentos")
    
    logger.info(f"URLs de descarga obtenidas para {len(video_urls)} segmentos")
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    video_paths = []
    output_path = f"{temp_dir}/concatenated.mp4"
    
    try:
        # Descargar todos los segmentos
        for i, (url, result) in enumerate(zip(video_urls, segment_results)):
            segment_path = f"{temp_dir}/segment{i+1}.mp4"
            logger.info(f"Descargando segmento {i+1}/{len(video_urls)}...")
            download_video_segment(url, segment_path, result)
            video_paths.append(segment_path)
        
        # Concatenar videos
        concatenate_videos_ffmpeg(video_paths, output_path)
        
        # Subir a GCS si está configurado
        if settings.gcs_bucket_name:
            logger.info("Subiendo video concatenado a GCS...")
            
            final_uri = upload_concatenated_to_gcs(
                output_path, 
                settings.gcs_bucket_name,
                settings.gcs_output_path
            )
            
            if final_uri:
                # Agregar al caché
                concat_operation_id = f"concat_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                add_to_cache(concat_operation_id, ConcatenatedVideo(final_uri))
                
                # Generar URL firmada
                try:
                    signed_url = generate_signed_url(final_uri)
                    logger.info("URL firmada generada para video concatenado")
                except Exception as url_error:
                    logger.warning(f"Error generando URL firmada: {url_error}")
                    signed_url = gcs_uri_to_https_url(final_uri)
                    logger.info("Usando URL pública para video concatenado")
                
                return _build_success_response(
                    operation_id=concat_operation_id,
                    video_uri=signed_url,
                    segment_duration=segment_duration,
                    fps=fps,
                    aspect_ratio=aspect_ratio,
                    transition_style=transition_style,
                    motion_strength=motion_strength,
                    total_images=total_images,
                    video_urls=video_urls,
                    num_segments=len(segment_results),
                    is_gcs=True
                )
            else:
                # Fallback a video local
                return _handle_local_fallback(
                    output_path, temp_dir, segment_duration, fps, aspect_ratio,
                    transition_style, motion_strength, total_images, video_urls,
                    len(segment_results), "Error en upload a GCS"
                )
        else:
            # Sin GCS configurado - devolver video local
            return _handle_local_fallback(
                output_path, temp_dir, segment_duration, fps, aspect_ratio,
                transition_style, motion_strength, total_images, video_urls,
                len(segment_results), "GCS no configurado"
            )
            
    finally:
        # Limpiar archivos temporales
        _cleanup_temp_dir(temp_dir)


def _build_success_response(
    operation_id: str,
    video_uri: str,
    segment_duration: int,
    fps: int,
    aspect_ratio: str,
    transition_style: str,
    motion_strength: float,
    total_images: int,
    video_urls: List[str],
    num_segments: int,
    is_gcs: bool = True
) -> Dict[str, Any]:
    """Construye respuesta de éxito para video concatenado"""
    return {
        "operation_id": operation_id,
        "video_uri": video_uri,
        "duration_seconds": segment_duration * num_segments,
        "concatenated": True,
        "image_count": total_images,
        "transitions": [
            {"type": "dynamic", "duration": 1.5, "description": "Transición dinámica entre segmentos"},
            {"type": "smooth", "duration": 1.0, "description": "Transición suave al final"}
        ],
        "message": f"Video de {segment_duration * num_segments}s generado exitosamente mediante concatenación",
        "metadata": {
            "resolution": "1080p",
            "fps": fps,
            "aspect_ratio": aspect_ratio,
            "transition_style": transition_style,
            "motion_strength": motion_strength,
            "segments_count": num_segments,
            "concatenation_method": "ffmpeg"
        },
        "segments": [
            {"uri": url, "duration": segment_duration}
            for url in video_urls
        ],
        "status": "completed"
    }


def _handle_local_fallback(
    output_path: str,
    temp_dir: str,
    segment_duration: int,
    fps: int,
    aspect_ratio: str,
    transition_style: str,
    motion_strength: float,
    total_images: int,
    video_urls: List[str],
    num_segments: int,
    reason: str
) -> Dict[str, Any]:
    """Maneja el fallback a video local cuando GCS falla"""
    logger.warning(f"Fallback: devolviendo video local ({reason})")
    
    local_video_size = os.path.getsize(output_path)
    logger.info(f"Video concatenado disponible localmente: {local_video_size} bytes")
    
    # Mover a ubicación más permanente
    permanent_path = f"/tmp/concatenated_video_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
    shutil.copy2(output_path, permanent_path)
    
    # Agregar al caché
    concat_operation_id = f"concat_local_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    add_to_cache(concat_operation_id, LocalConcatenatedVideo(permanent_path))
    logger.info(f"Video concatenado local añadido al caché: {concat_operation_id}")
    
    response = _build_success_response(
        operation_id=concat_operation_id,
        video_uri=f"file://{permanent_path}",
        segment_duration=segment_duration,
        fps=fps,
        aspect_ratio=aspect_ratio,
        transition_style=transition_style,
        motion_strength=motion_strength,
        total_images=total_images,
        video_urls=video_urls,
        num_segments=num_segments,
        is_gcs=False
    )
    
    response["local_file"] = True
    response["metadata"]["local_storage"] = True
    response["note"] = f"Video disponible localmente ({reason})"
    
    return response


def _cleanup_temp_dir(temp_dir: str) -> None:
    """Limpia el directorio temporal de forma robusta"""
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Directorio temporal eliminado: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error limpiando archivos temporales en {temp_dir}: {e}")
