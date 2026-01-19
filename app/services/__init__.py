# Services module exports
from app.services.gemini_service import GeminiService

# Prompt optimization functions
from app.services.prompt_optimizer import (
    enhance_prompt_consistency,
    detect_language,
    optimize_prompt_for_reel,
    get_text_overlay_specifications,
    validate_text_readability_prompt,
    get_narration_consistency_specs,
    build_reel_enhanced_prompt,
    build_showcase_enhanced_prompt,
    get_final_restrictions,
    build_extension_prompt
)

# Cache management functions
from app.services.cache_service import (
    get_video_cache,
    add_to_cache,
    get_from_cache,
    remove_from_cache,
    clear_cache,
    save_cache_to_disk,
    load_cache_from_disk,
    get_cache_stats,
    ConcatenatedVideo,
    LocalConcatenatedVideo
)

# Storage service functions (GCS)
from app.services.storage_service import (
    download_video_from_gcs,
    generate_signed_url,
    generate_gcs_output_uri,
    gcs_uri_to_https_url,
    upload_video_to_gcs,
    get_storage_client,
    check_blob_exists
)

# Video processing functions (FFmpeg)
from app.services.video_processor import (
    check_ffmpeg_available,
    download_video_segment,
    concatenate_videos_ffmpeg,
    get_video_url_from_result,
    process_concatenated_video
)

__all__ = [
    # Main service
    'GeminiService',
    
    # Prompt optimization
    'enhance_prompt_consistency',
    'detect_language',
    'optimize_prompt_for_reel',
    'get_text_overlay_specifications',
    'validate_text_readability_prompt',
    'get_narration_consistency_specs',
    'build_reel_enhanced_prompt',
    'build_showcase_enhanced_prompt',
    'get_final_restrictions',
    'build_extension_prompt',
    
    # Cache management
    'get_video_cache',
    'add_to_cache',
    'get_from_cache',
    'remove_from_cache',
    'clear_cache',
    'save_cache_to_disk',
    'load_cache_from_disk',
    'get_cache_stats',
    'ConcatenatedVideo',
    'LocalConcatenatedVideo',
    
    # Storage service (GCS)
    'download_video_from_gcs',
    'generate_signed_url',
    'generate_gcs_output_uri',
    'gcs_uri_to_https_url',
    'upload_video_to_gcs',
    'get_storage_client',
    'check_blob_exists',
    
    # Video processing (FFmpeg)
    'check_ffmpeg_available',
    'download_video_segment',
    'concatenate_videos_ffmpeg',
    'get_video_url_from_result',
    'process_concatenated_video'
]
