"""
Tests para el servicio de generación de videos con Veo
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.gemini_service import GeminiService


@pytest.fixture
def gemini_service():
    """Fixture para crear una instancia del servicio Gemini"""
    return GeminiService(api_key="test-api-key")


@pytest.mark.asyncio
async def test_generate_video_basic(gemini_service):
    """Test básico de generación de video"""
    # Mock del cliente y la operación
    mock_operation = Mock()
    mock_operation.done = True
    mock_operation.name = "test-operation-123"
    mock_operation.response = Mock()
    mock_operation.response.generated_videos = [Mock()]
    mock_operation.response.generated_videos[0].video = Mock()
    mock_operation.response.generated_videos[0].video.uri = "gs://test/video.mp4"
    mock_operation.response.usage_metadata = Mock()
    mock_operation.response.usage_metadata.prompt_token_count = 15
    mock_operation.response.usage_metadata.video_token_count = 1000
    mock_operation.response.usage_metadata.total_token_count = 1015
    
    with patch.object(gemini_service.client.models, 'generate_videos', return_value=mock_operation):
        with patch.object(gemini_service.client.operations, 'get', return_value=mock_operation):
            result = await gemini_service.generate_video(
                prompt="Un gato jugando en el jardín",
                model_name="veo-3.1-generate-preview"
            )
            
            assert result["operation_id"] == "test-operation-123"
            assert result["video_uri"] == "gs://test/video.mp4"
            assert result["duration_seconds"] == 8
            assert result["resolution"] == "720p"
            assert result["aspect_ratio"] == "16:9"
            assert result["usage"]["prompt_tokens"] == 15
            assert result["usage"]["video_tokens"] == 1000
            assert result["usage"]["total_tokens"] == 1015


@pytest.mark.asyncio
async def test_generate_video_with_reference_images(gemini_service):
    """Test de generación de video con imágenes de referencia"""
    mock_operation = Mock()
    mock_operation.done = True
    mock_operation.name = "test-operation-456"
    mock_operation.response = Mock()
    mock_operation.response.generated_videos = [Mock()]
    mock_operation.response.generated_videos[0].video = Mock()
    
    # Mock para el procesamiento de imágenes
    with patch.object(gemini_service, '_process_image_input', return_value=Mock()):
        with patch.object(gemini_service.client.models, 'generate_videos', return_value=mock_operation):
            with patch.object(gemini_service.client.operations, 'get', return_value=mock_operation):
                result = await gemini_service.generate_video(
                    prompt="Una mujer caminando en la playa",
                    reference_images=["https://example.com/image1.jpg", "data:image/jpeg;base64,test"],
                    aspect_ratio="9:16",
                    resolution="1080p",
                    duration_seconds=6
                )
                
                assert result["operation_id"] == "test-operation-456"
                assert result["aspect_ratio"] == "9:16"
                assert result["resolution"] == "1080p"
                assert result["duration_seconds"] == 6


@pytest.mark.asyncio
async def test_generate_video_error_handling(gemini_service):
    """Test de manejo de errores en generación de video"""
    with patch.object(gemini_service.client.models, 'generate_videos', side_effect=Exception("API Error")):
        with pytest.raises(Exception) as exc_info:
            await gemini_service.generate_video("Test prompt")
        
        assert "Error al generar video con Veo" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_video_timeout(gemini_service):
    """Test de timeout en generación de video"""
    mock_operation = Mock()
    mock_operation.done = False  # Nunca se completa
    mock_operation.name = "test-operation-timeout"
    
    with patch.object(gemini_service.client.models, 'generate_videos', return_value=mock_operation):
        with patch.object(gemini_service.client.operations, 'get', return_value=mock_operation):
            with patch('time.time', side_effect=[0, 700]):  # Simular timeout
                with pytest.raises(Exception) as exc_info:
                    await gemini_service.generate_video("Test prompt")
                
                assert "Tiempo de espera agotado" in str(exc_info.value)


def test_process_image_input_url(gemini_service):
    """Test de procesamiento de imagen desde URL"""
    mock_response = Mock()
    mock_response.content = b"fake_image_data"
    
    with patch('requests.get', return_value=mock_response):
        with patch('PIL.Image.open') as mock_image:
            result = gemini_service._process_image_input("https://example.com/image.jpg")
            mock_image.assert_called_once()


def test_process_image_input_base64(gemini_service):
    """Test de procesamiento de imagen desde base64"""
    base64_data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD"
    
    with patch('base64.b64decode', return_value=b"fake_image_data"):
        with patch('PIL.Image.open') as mock_image:
            result = gemini_service._process_image_input(base64_data)
            mock_image.assert_called_once()


def test_process_image_input_error(gemini_service):
    """Test de error en procesamiento de imagen"""
    with patch('requests.get', side_effect=Exception("Network error")):
        with pytest.raises(Exception) as exc_info:
            gemini_service._process_image_input("https://invalid-url.com/image.jpg")
        
        assert "Error al procesar imagen" in str(exc_info.value)


@pytest.mark.asyncio
async def test_download_video(gemini_service):
    """Test de descarga de video"""
    mock_video_file = Mock()
    mock_video_data = b"fake_video_data"
    
    with patch.object(gemini_service.client.files, 'download', return_value=mock_video_data):
        result = await gemini_service.download_video(mock_video_file, "test_video.mp4")
        assert result == mock_video_data


@pytest.mark.asyncio
async def test_download_video_error(gemini_service):
    """Test de error en descarga de video"""
    mock_video_file = Mock()
    
    with patch.object(gemini_service.client.files, 'download', side_effect=Exception("Download error")):
        with pytest.raises(Exception) as exc_info:
            await gemini_service.download_video(mock_video_file)
        
        assert "Error al descargar video" in str(exc_info.value)


def test_get_available_models_includes_video_models(gemini_service):
    """Test que verifica que los modelos de video están incluidos"""
    models = gemini_service.get_available_models()
    
    # Verificar que incluye modelos de video
    video_models = [model for model in models if 'veo' in model]
    assert len(video_models) > 0
    assert "veo-3.1-generate-preview" in models
    assert "veo-3.1-fast-preview" in models