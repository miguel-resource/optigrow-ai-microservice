"""
Módulo de optimización de prompts para generación de videos

Este módulo contiene todas las funciones relacionadas con:
- Optimización de prompts para estética de reel/Instagram
- Especificaciones de texto overlay legible
- Consistencia de narración e idioma
- Coherencia entre segmentos de video

Separado de gemini_service.py para mejor organización y mantenibilidad.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def enhance_prompt_consistency(
    base_prompt: str, 
    segment_number: int, 
    total_segments: int, 
    previous_context: str = "", 
    dynamic_camera_changes: bool = True, 
    maintain_language: bool = True
) -> str:
    """
    Mejora la consistencia del prompt entre segmentos de video
    
    Args:
        base_prompt: Prompt base del usuario
        segment_number: Número del segmento actual (1-based)
        total_segments: Total de segmentos planificados
        previous_context: Contexto del segmento anterior para coherencia
        dynamic_camera_changes: Incluir cambios dinámicos de cámara
        maintain_language: Mantener consistencia de idioma entre segmentos
        
    Returns:
        Prompt mejorado para mantener coherencia entre segmentos
    """
    try:
        # Prefijos según la posición del segmento
        if segment_number == 1:
            position_prefix = "Opening sequence: "
            camera_style = "establishing shot, wide angle, product introduction"
        elif segment_number == total_segments:
            position_prefix = "Closing sequence: "
            camera_style = "close-up details, final showcase, call-to-action angle"
        else:
            position_prefix = f"Continuation sequence {segment_number}/{total_segments}: "
            camera_style = "dynamic angle change, different perspective, engaging transition"
        
        # Elementos de coherencia
        consistency_elements = [
            "maintaining visual style",
            "consistent lighting",
            "smooth narrative flow",
            "matching color palette",
            "seamless transitions",
            "continuous narration voice",
            "SAME LANGUAGE throughout",
            "LEGIBLE text overlays with high contrast",
            "readable typography throughout",
            "NO language switching"
        ]
        
        # Construir prompt mejorado
        enhanced_prompt = f"{position_prefix}{base_prompt}"
        
        if previous_context:
            enhanced_prompt += f" Continuing from: {previous_context}"
        
        if dynamic_camera_changes:
            enhanced_prompt += f". Camera: {camera_style}"
        
        if maintain_language:
            # Forzar siempre español para todos los videos
            detected_language = "spanish"
            enhanced_prompt += f". MAINTAIN {detected_language.upper()} ONLY, NO language mixing"
        
        enhanced_prompt += f". Ensure: {', '.join(consistency_elements[:6])}, CLEAR READABLE text overlays, continuous narration WITHOUT REPETITIONS"
        
        logger.info(f"Prompt mejorado para segmento {segment_number}/{total_segments}")
        return enhanced_prompt
        
    except Exception as e:
        logger.warning(f"Error mejorando consistencia del prompt: {e}")
        return base_prompt


def detect_language(text: str) -> str:
    """
    Detecta el idioma del texto - siempre devuelve español para garantizar contenido en español
    
    Args:
        text: Texto a analizar
        
    Returns:
        Siempre 'spanish' para asegurar contenido en español
    """
    # Forzar siempre español para todos los videos
    return "spanish"


def optimize_prompt_for_reel(
    prompt: str, 
    is_product_showcase: bool = True, 
    add_narration: bool = True, 
    text_overlays: bool = True, 
    language: str = "spanish"
) -> str:
    """
    Optimiza el prompt para crear contenido natural tipo reel con personas reales
    
    Args:
        prompt: Prompt original del usuario
        is_product_showcase: Si es un showcase de producto
        add_narration: Incluir elementos de narración
        text_overlays: Incluir texto overlays dinámicos
        language: Idioma para mantener consistencia (default: spanish)
        
    Returns:
        Prompt optimizado para reel natural con personas
    """
    try:
        # Prompt base para reel natural con personas reales
        base_reel_prompt = f"REEL NATURAL VERTICAL 9:16: {prompt}. "
        
        # Elementos de autenticidad y naturalidad
        authenticity_elements = [
            "Persona real (NO AI obvio) demostrando el producto auténticamente",
            "Iluminación natural, ambiente real",
            "Movimientos de cámara SUAVES y naturales",
            "Audio consistente con volumen estable",
            "Expresiones genuinas y creíbles"
        ]
        
        # Elementos específicos para showcase de producto
        if is_product_showcase:
            product_elements = [
                "Mostrar EXACTAMENTE el producto de las imágenes",
                "NO cambiar características o inventar funciones",
                "Mantener colores y detalles reales del producto",
                "Enfocar en beneficios visibles y reales",
                "Evitar ángulos dramáticos si las imágenes son limitadas"
            ]
        else:
            product_elements = []
        
        # Elementos de narración natural con consistencia específica
        narration_elements = []
        if add_narration:
            narration_elements = [
                f"UNA SOLA persona hablando consistentemente en {language}",
                "MISMO narrador durante TODO el video y extensiones", 
                "Voz característica y única que se mantenga en extensiones",
                "Tono conversacional, auténtico y CONSTANTE",
                "Explicación clara y directa de beneficios del producto",
                f"Audio limpio y consistente en {language}",
                "NO cambios de narrador o presentador",
                "Mantener MISMA personalidad vocal en extensiones",
                "Ritmo natural y estable de conversación"
            ]
        
        # Elementos de texto overlay minimalista
        text_elements = []
        if text_overlays:
            text_elements = [
                f"Texto overlay MINIMALISTA en {language}",
                "Máximo 1-2 palabras clave",
                "Fuente bold, alto contraste",
                "Aparición sutil y breve",
                "Coordinado con la narración"
            ]
        
        # Restricciones críticas para consistencia de narrador y producto
        consistency_rules = [
            "MANTENER el mismo narrador en toda extensión (MISMA voz y persona)",
            "MANTENER el mismo producto en toda extensión",
            "PROHIBIDO cambiar de presentador o voz entre segmentos",
            "PROHIBIDO cambiar ángulos dramáticamente",
            "PROHIBIDO inventar características no visibles",
            "PROHIBIDO cambios abruptos de volumen o tono",
            "MANTENER el mismo estilo visual y vocal",
            "Continuidad total de la misma persona hablando"
        ]
        
        # Combinar elementos según configuración
        all_elements = authenticity_elements
        if product_elements:
            all_elements.extend(product_elements[:3])
        if narration_elements:
            all_elements.extend(narration_elements[:3])
        if text_elements:
            all_elements.extend(text_elements[:2])
        
        # Construir prompt optimizado
        optimized_prompt = f"{base_reel_prompt}"
        optimized_prompt += f"CARACTERÍSTICAS: {'. '.join(all_elements[:8])}. "
        optimized_prompt += f"REGLAS: {'. '.join(consistency_rules[:3])}."
        
        logger.info(f"Prompt optimizado para reel natural: {len(optimized_prompt)} caracteres")
        return optimized_prompt
        
    except Exception as e:
        logger.warning(f"Error optimizando prompt: {e}")
        return prompt


def get_text_overlay_specifications(aspect_ratio: str = "9:16") -> str:
    """
    Genera especificaciones detalladas para text overlays legibles
    
    Args:
        aspect_ratio: Formato del video para optimizar posicionamiento
        
    Returns:
        Especificaciones técnicas para texto legible
    """
    try:
        if aspect_ratio == "9:16":
            # Optimizado para móviles verticales con texto MINIMALISTA
            specs = {
                "font_size": "EXTRA LARGE BOLD (minimum 32pt equivalent)",
                "position": "upper third or lower third, never center over product",
                "contrast": "MAXIMUM CONTRAST: Pure WHITE text on SOLID BLACK background OR pure BLACK text on BRIGHT WHITE background",
                "duration": "2-3 seconds per text element (short and impactful)",
                "animation": "smooth fade-in/fade-out, no jarring movements",
                "content": "ONLY 1-2 KEY WORDS extracted from narration"
            }
        else:
            # Para otros formatos con texto minimalista
            specs = {
                "font_size": "LARGE BOLD (minimum 28pt equivalent)",
                "position": "corners or sides, avoiding product center",
                "contrast": "ULTRA HIGH contrast color combinations",
                "duration": "2-3 seconds for quick readability",
                "animation": "gentle transitions",
                "content": "MINIMAL text: 1-2 words maximum"
            }
        
        specifications = f"""
        TEXT OVERLAY SPECIFICATIONS (MINIMALIST APPROACH):
        - Font Size: {specs['font_size']}
        - Position: {specs['position']}
        - Contrast: {specs['contrast']}
        - Duration: {specs['duration']}
        - Animation: {specs['animation']}
        - Content: {specs['content']}
        - Style: Bold, clean, professional sans-serif font
        - Language: SAME as narration, NO mixing
        - Examples: "RENDIMIENTO", "CALIDAD", "MEJORA" (single impactful words)
        - NEVER use full sentences or complex phrases
        - Extract EXACT key words from what narrator says
        """
        
        return specifications
        
    except Exception as e:
        logger.warning(f"Error generando especificaciones de texto: {e}")
        return "Use large, high-contrast, readable text overlays"


def validate_text_readability_prompt(prompt: str, aspect_ratio: str = "9:16") -> str:
    """
    Valida y mejora un prompt para asegurar texto legible
    
    Args:
        prompt: Prompt original
        aspect_ratio: Formato del video
        
    Returns:
        Prompt mejorado con especificaciones de legibilidad
    """
    try:
        # Palabras clave que indican problemas potenciales de legibilidad
        problematic_words = ['small', 'tiny', 'subtle', 'faint', 'light', 'transparent']
        improved_prompt = prompt
        
        # Reemplazar términos problemáticos
        for word in problematic_words:
            if word in improved_prompt.lower():
                if 'text' in improved_prompt.lower():
                    improved_prompt = improved_prompt.replace(word, 'LARGE BOLD')
                    logger.info(f"Reemplazado '{word}' con 'LARGE BOLD' para mejor legibilidad")
        
        # Agregar especificaciones específicas si no están presentes
        readability_keywords = ['contrast', 'legible', 'readable', 'clear', 'bold']
        if not any(keyword in improved_prompt.lower() for keyword in readability_keywords):
            if aspect_ratio == "9:16":
                improved_prompt += " WITH MAXIMUM CONTRAST LARGE TEXT optimized for mobile viewing"
            else:
                improved_prompt += " WITH HIGH CONTRAST READABLE TEXT"
            logger.info("Agregadas especificaciones de legibilidad al prompt")
        
        return improved_prompt
        
    except Exception as e:
        logger.warning(f"Error validando legibilidad del prompt: {e}")
        return prompt


def get_narration_consistency_specs(language: str = "spanish") -> str:
    """
    Genera especificaciones para narración consistente sin cambios de idioma ni repeticiones
    
    Args:
        language: Idioma objetivo para la narración
        
    Returns:
        Especificaciones para narración de calidad
    """
    try:
        language_specs = {
            "spanish": {
                "voice": "clear professional Spanish voice",
                "pronunciation": "perfect Spanish pronunciation", 
                "consistency": "ONLY Spanish throughout entire video",
                "restrictions": "NO English, NO other languages, NO mixed speech"
            },
            "english": {
                "voice": "clear professional English voice",
                "pronunciation": "perfect English pronunciation",
                "consistency": "ONLY English throughout entire video", 
                "restrictions": "NO Spanish, NO other languages, NO mixed speech"
            }
        }
        
        specs = language_specs.get(language.lower(), language_specs["spanish"])
        
        narration_specifications = f"""
        NARRATION CONSISTENCY SPECIFICATIONS:
        - Voice: {specs['voice']}
        - Pronunciation: {specs['pronunciation']}
        - Language Consistency: {specs['consistency']}
        - Speech Quality: Smooth, natural flow without repetitions
        - Restrictions: {specs['restrictions']}
        - NO stuttering, NO repeated letters (like "p-p-producto")
        - NO language switching mid-sentence or mid-video
        - Maintain same voice actor and tone throughout
        - Clear articulation without verbal tics
        """
        
        return narration_specifications
        
    except Exception as e:
        logger.warning(f"Error generando especificaciones de narración: {e}")
        return f"Use consistent {language} narration throughout entire video without language switching"


def build_reel_enhanced_prompt(
    prompt: str,
    duration_seconds: int,
    num_images: int,
    transition_style: str,
    time_per_image: float,
    detected_language: str,
    dynamic_camera_changes: bool = True
) -> str:
    """
    Construye el prompt completo optimizado para contenido tipo reel
    
    Args:
        prompt: Prompt base optimizado
        duration_seconds: Duración del video
        num_images: Número de imágenes
        transition_style: Estilo de transición
        time_per_image: Tiempo por imagen
        detected_language: Idioma (forzado a español para todos los videos)
        dynamic_camera_changes: Si usar cambios de cámara dinámicos
        
    Returns:
        Prompt completo para generación de reel en español
    """
    # Forzar siempre español para todos los videos
    detected_language = "spanish"
    enhanced_prompt = f"""
    REEL PROFESIONAL: Crear video {duration_seconds}s estilo Instagram/TikTok optimizado para móviles
    usando estas {num_images} imágenes del producto. {prompt}
    
    ESTILO REEL OPTIMIZADO:
    - Estética moderna y atractiva para redes sociales
    - Movimientos cinematográficos suaves y profesionales  
    - Iluminación y contraste optimizados para dispositivos móviles
    - Transiciones fluidas {transition_style} que mantengan engagement ({time_per_image:.1f}s por imagen)
    - Audio continuo y sincronizado sin cortes ni distorsión
    - Composición vertical perfecta para formato 9:16
    - CÁMARA DINÁMICA: {"Cambios de ángulo y perspectiva" if dynamic_camera_changes else "Movimientos suaves y constantes"}
    - NARRACIÓN CONTINUA EN {detected_language.upper()}: Voz envolvente que explique el producto durante todo el video
    - TEXTO MINIMALISTA EN {detected_language.upper()}: 
      * SOLO 1-2 PALABRAS CLAVE por overlay
      * Extraer PALABRAS EXACTAS de lo que dice el narrador
      * Ejemplo: si narrador dice "Mejora tu rendimiento" → mostrar solo "RENDIMIENTO" o "MEJORA"
      * MÁXIMO CONTRASTE: Texto blanco puro sobre fondo negro sólido o viceversa
      * Tamaño EXTRA GRANDE y BOLD para móviles (mínimo 32pt equivalente)
      * Tipografía ultra clara (Arial Black, Helvetica Bold)
      * Posición estratégica que no interfiera con el producto
      * Duración corta e impactante (2-3 segundos por palabra)
      * Animaciones suaves de entrada y salida
      * NO usar frases completas, SOLO palabras clave
      * MISMO idioma {detected_language.upper()} que la narración
    - ELEMENTOS VISUALES: Palabras clave destacadas, mensajes simples e impactantes
    
    COHERENCIA Y CONTINUIDAD:
    - MANTENER tema principal "{prompt}" en TODO el video
    - Audio limpio y continuo entre segmentos (sin cortes abruptos)
    - Cada segmento conecta naturalmente con el anterior
    - Preservar mensaje central y estilo visual consistente
    - Transiciones suaves que no rompan el flujo narrativo
    - IDIOMA CONSISTENTE: Solo {detected_language.upper()} durante todo el video
    - NARRACIÓN SIN REPETICIONES: Pronunciación clara y fluida
    - TEXTO MINIMALISTA: Solo palabras clave que coincidan con narración
    - NO cambiar de idioma en ningún momento del video
    - NO usar texto en otros idiomas o texto confuso
    
    FIDELIDAD DEL PRODUCTO:
    - Mostrar EXACTAMENTE el producto de las imágenes proporcionadas
    - Mantener características, colores, forma y detalles específicos
    - NO inventar elementos que no estén en las imágenes originales
    - Aplicar efectos sin distorsionar la apariencia real del producto
    """
    
    return enhanced_prompt


def build_showcase_enhanced_prompt(
    prompt: str,
    duration_seconds: int,
    num_images: int,
    transition_style: str,
    time_per_image: float,
    dynamic_camera_changes: bool = True
) -> str:
    """
    Construye el prompt completo para showcase profesional (no reel)
    
    Args:
        prompt: Prompt base optimizado
        duration_seconds: Duración del video
        num_images: Número de imágenes
        transition_style: Estilo de transición
        time_per_image: Tiempo por imagen
        dynamic_camera_changes: Si usar cambios de cámara dinámicos
        
    Returns:
        Prompt completo para showcase profesional
    """
    enhanced_prompt = f"""
    SHOWCASE PROFESIONAL: Crear video promocional de {duration_seconds} segundos estilo reel 
    usando estas {num_images} imágenes del producto. {prompt}
    
    AUDIO Y ESTÉTICA:
    - Audio continuo y sin cortes para mejor experiencia 
    - NARRACIÓN PROFESIONAL EN ESPAÑOL: Voz clara que describe el producto durante todo el video ÚNICAMENTE en español
    - TEXTO MINIMALISTA EN ESPAÑOL (1-2 PALABRAS MÁXIMO):
      * Usar SOLO palabras clave extraídas de la descripción del producto EN ESPAÑOL
      * Ejemplo: si el producto es "Proteína de alta calidad" → mostrar solo "PROTEÍNA" o "CALIDAD"
      * MÁXIMO CONTRASTE: Texto blanco puro sobre negro sólido o viceversa
      * Fuente EXTRA GRANDE y BOLD (mínimo 32pt) para móviles
      * Texto ultra NÍTIDO sin pixelación
      * Posicionamiento que no obstruya el producto
      * Palabras impactantes y simples EN ESPAÑOL
      * SOLO ESPAÑOL en toda la narración y texto
      * NO usar frases completas, SOLO palabras clave en español
      * Duración corta: 2-3 segundos por palabra
    - MOVIMIENTO DE CÁMARA: {"Dinámico con cambios de perspectiva" if dynamic_camera_changes else "Suave y consistente"}
    - Calidad profesional optimizada para redes sociales
    - Transiciones suaves que mantengan engagement
    - Movimientos cinematográficos controlados
    
    FIDELIDAD DEL PRODUCTO: 
    Mantener EXACTAMENTE las características, colores, forma y textura del producto real.
    NO inventar ni modificar elementos. Mostrar únicamente lo que aparece en las imágenes.
    
    Transiciones {transition_style} optimizadas para reel ({time_per_image:.1f}s por imagen).
    Audio sincronizado y de alta calidad.
    """
    
    return enhanced_prompt


def get_final_restrictions(dynamic_camera_changes: bool = True) -> str:
    """
    Genera las restricciones finales para el prompt
    
    Args:
        dynamic_camera_changes: Si usar cambios de cámara dinámicos
        
    Returns:
        String con restricciones finales
    """
    return f"""

COHERENCIA, NARRACIÓN Y AUDIO OPTIMIZADOS:
- Mantener continuidad visual y narrativa perfecta
- NARRACIÓN CONTINUA: Voz profesional que no se corte entre segmentos
- IDIOMA CONSISTENTE: NO cambiar de idioma durante el video
- PRONUNCIACIÓN PERFECTA: Sin repeticiones de letras ni tartamudeo
- TEXTO DINÁMICO LEGIBLE: Overlays de alta calidad, nítidos y fáciles de leer
- Audio limpio, sincronizado y sin distorsión
- Estética profesional de reel/social media
- Transiciones que mejoren el engagement
- CAMBIOS DE CÁMARA: {'Dinámicos en extensiones y transiciones' if dynamic_camera_changes else 'Suaves y consistentes'}

RESTRICCIONES DE FIDELIDAD Y LEGIBILIDAD:
- NO crear, inventar o añadir elementos que no estén en las imágenes
- NO modificar forma, color, textura o características del producto
- NO usar texto pequeño, borroso o de bajo contraste
- NO cambiar de idioma durante el video
- NO repetir letras o palabras en la narración
- Mantener absoluta fidelidad al producto real de las imágenes
- TEXTO SIEMPRE LEGIBLE: Alto contraste, tamaño adecuado, tipografía clara
- NARRACIÓN SIEMPRE CONSISTENTE: Mismo idioma, misma voz, pronunciación perfecta
- Priorizar calidad del showcase y narración sobre efectos creativos
- Audio continuo de alta calidad con narración envolvente
- Texto overlays NÍTIDOS, contrastados y profesionales"""


def get_localized_text_specs(detected_language: str) -> str:
    """
    Genera especificaciones de texto overlay siempre en español
    
    Args:
        detected_language: Idioma (ignorado, siempre usa español)
        
    Returns:
        Especificaciones de texto en español
    """
    # Forzar siempre español para todos los videos
    return f"""
    REGLAS DE TEXTO OVERLAY PARA EXTENSIÓN (SIEMPRE EN ESPAÑOL):
    - Usar ÚNICAMENTE español para TODO el texto
    - Extraer PALABRAS CLAVE CORTAS de la narración (máximo 1-2 palabras)
    - Ejemplos: si la narración dice "Mejora tu rendimiento" → mostrar solo "RENDIMIENTO" o "MEJORA"
    - Usar palabras EXACTAS de la descripción del producto que menciona el narrador
    - Texto MINIMALISTA: preferir palabras impactantes individuales
    - MÁXIMO 2 palabras por texto overlay
    - ULTRA ALTO CONTRASTE: Texto blanco sobre fondo negro o viceversa
    - Tamaño de fuente EXTRA GRANDE para legibilidad móvil
    - Mostrar texto solo por 2-3 segundos
    - NO usar frases complejas, NO oraciones completas
    - NO mezclar idiomas, NO texto en inglés
    - TODO EL CONTENIDO DEBE SER EN ESPAÑOL
    """


def build_extension_prompt(
    core_theme: str,
    detected_language: str,
    dynamic_camera_changes: bool = False,  # Cambiar default a False
    is_reel_content: bool = True
) -> str:
    """
    Construye el prompt para extensión de video manteniendo máxima consistencia
    
    Args:
        core_theme: Tema principal del video
        detected_language: Idioma (ignorado, siempre usa español)
        dynamic_camera_changes: Si usar cambios de cámara dinámicos (False por defecto)
        is_reel_content: Si es contenido tipo reel
        
    Returns:
        Prompt para extensión que mantiene total consistencia
    """
    
    # Prompt base que garantiza continuidad absoluta
    base_prompt = f"""CONTINUAR VIDEO EXACTO - MISMO PRODUCTO: {core_theme}

REGLAS ABSOLUTAS PARA EXTENSIÓN:
1. MISMO PRODUCTO: Continuar con el producto EXACTO mostrado en el video anterior
2. MISMO NARRADOR: La MISMA persona que habló en el video anterior debe continuar hablando
3. MISMA VOZ: Mantener exactamente la misma voz, tono, acento y estilo de narración
4. MISMO CONTEXTO: Mantener la misma persona, ambiente y escenario visual
5. MISMO VOLUMEN: Audio consistente, SIN cambios de volumen abruptos
6. MISMA PERSPECTIVA: NO cambiar ángulos dramáticamente
7. MISMO TEMA: NO introducir productos o temas diferentes

CONTINUIDAD DE NARRACIÓN:
- El MISMO narrador que comenzó el video debe continuar hasta el final
- Mantener exactamente el MISMO tono de voz y estilo narrativo
- Continuar la conversación donde se quedó en el video anterior
- NO cambiar de persona que habla o presenta el producto
- Usar las MISMAS características vocales (velocidad, entonación, personalidad)

CONTINUACIÓN NATURAL:
- La MISMA persona continúa explicando el MISMO producto
- Mantener el mismo ritmo y energía del video base
- Seguir el flujo natural de la presentación anterior
- Enfocarse en características ya mencionadas o visibles
- NO inventar nuevas funciones o cambiar el producto

PROHIBIDO EN EXTENSIÓN:
- Cambiar de narrador o persona que habla
- Usar una voz diferente o nuevo presentador
- Cambiar el estilo de narración establecido
- Modificar el tono o personalidad de la voz
- Cambiar de producto o mostrar productos diferentes
- Cambiar dramáticamente el volumen de audio
- Introducir nuevos escenarios o personas
- Modificar el estilo visual establecido
- Alucinar características no presentes en imágenes originales

ESTILO REEL NATURAL:
- Persona real demostrando el producto
- Iluminación natural y consistente
- Movimientos de cámara MÍNIMOS y suaves
- Audio con nivel constante
- Transiciones naturales y sutiles"""

    # Agregar especificaciones de cámara según el parámetro
    if dynamic_camera_changes and is_reel_content:
        camera_specs = """
MOVIMIENTO DE CÁMARA SUTIL:
- Zoom muy gradual en detalles específicos
- Movimiento lateral LENTO si es necesario
- Mantener el mismo encuadre general
- NO hacer movimientos bruscos o dramáticos"""
    else:
        camera_specs = """
CÁMARA ESTÁTICA:
- Mantener el mismo encuadre del video anterior
- Solo pequeños ajustes de enfoque si es necesario
- Evitar cualquier movimiento dramático
- Preservar la misma perspectiva visual"""

    # Especificaciones de texto overlay y audio
    text_specs = """
TEXTO OVERLAY CONSISTENTE:
- Máximo 1-2 palabras relacionadas al producto
- Fuente bold, alto contraste
- Aparición sutil y coherente con la narración
- TODO EN ESPAÑOL, texto minimalista
- NO usar frases completas o texto complejo

CONTROL DE AUDIO Y NARRACIÓN:
- MISMA voz del narrador original debe continuar
- Mantener exactamente el MISMO tono y estilo vocal
- Audio con nivel constante, sin cambios de volumen
- Continuar la conversación natural donde se quedó
- NO introducir nuevos narradores o voces diferentes
- Preservar las características únicas de la voz original"""

    return f"{base_prompt}\n{camera_specs}\n{text_specs}\n\nLa extensión debe ser una continuación perfecta y natural del video anterior."
