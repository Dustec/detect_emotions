import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from datetime import datetime, timedelta
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

# Configuración de la página
st.set_page_config(
    page_title="Detector de Emociones",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de constantes
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']
TEMP_DIR = tempfile.gettempdir()
RETENTION_HOURS = 1  # Eliminación automática después de 1 hora

# Tipología de emociones con emojis
EMOTION_LABELS = {
    0: {"name": "Angry", "emoji": "😠", "color": "#FF6B6B"},
    1: {"name": "Disgust", "emoji": "🤢", "color": "#A8E6CF"},
    2: {"name": "Fear", "emoji": "😨", "color": "#FFB347"},
    3: {"name": "Happy", "emoji": "😊", "color": "#FFD93D"},
    4: {"name": "Neutral", "emoji": "😐", "color": "#C7C7C7"},
    5: {"name": "Sad", "emoji": "😢", "color": "#87CEEB"},
    6: {"name": "Surprise", "emoji": "😲", "color": "#DDA0DD"}
}


@st.cache_resource
def load_emotion_model():
    """Carga el modelo entrenado para detección de emociones (evita errores de compatibilidad)"""
    try:
        model_path = "emotion_model.h5"

        # Evitar error de 'lr' usando compile=False
        model = load_model(model_path, compile=False)

        # Compilar manualmente (no importa el optimizador en inferencia)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo preentrenado: {str(e)}")
        return None


@st.cache_resource
def load_face_cascade():
    """Carga el clasificador de rostros de OpenCV"""
    try:
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        st.error(f"Error al cargar el detector de rostros: {str(e)}")
        return None


def validate_image_file(uploaded_file):
    """Valida el archivo de imagen subido"""
    if uploaded_file is None:
        return False, "No se ha seleccionado ningún archivo"

    # Validar extensión
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return False, f"Formato no permitido. Use: {', '.join(ALLOWED_EXTENSIONS)}"

    # Validar tamaño
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"Archivo muy grande. Máximo permitido: {MAX_FILE_SIZE // (1024*1024)}MB"

    return True, "Archivo válido"


def preprocess_face(face_img):
    """Preprocesa la imagen del rostro para el modelo preentrenado de 64x64"""
    try:
        # Convertir a escala de grises si tiene 3 canales
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Redimensionar a 64x64
        face_img = cv2.resize(face_img, (64, 64))

        # Normalizar
        face_img = face_img.astype('float32') / 255.0

        # Expandir dims
        face_img = np.expand_dims(face_img, axis=-1)  # (64, 64, 1)
        face_img = np.expand_dims(face_img, axis=0)   # (1, 64, 64, 1)

        return face_img
    except Exception as e:
        st.error(f"Error en preprocesamiento: {str(e)}")
        return None


def detect_emotion(image, model, face_cascade):
    """Detecta emociones en la imagen"""
    try:
        # Convertir PIL a OpenCV
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convertir RGB a BGR para OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Convertir a escala de grises para detección de rostros
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        results = []

        for (x, y, w, h) in faces:
            # Extraer rostro
            face_roi = gray[y:y+h, x:x+w]

            # Preprocesar
            processed_face = preprocess_face(face_roi)

            if processed_face is not None:
                prediction = model.predict(processed_face, verbose=0)
                prediction = prediction[0]  # shape: (7,)

                # Asegurar que prediction sea un array
                if isinstance(prediction, list):
                    prediction = np.array(prediction)

                emotion_idx = np.argmax(prediction)
                confidence = float(np.max(prediction))

                results.append({
                    'bbox': (x, y, w, h),
                    'emotion': emotion_idx,
                    'confidence': confidence,
                    'prediction': prediction
                })

        return results
    except Exception as e:
        st.error(f"Error en detección de emociones: {str(e)}")
        return []


def create_temp_file(uploaded_file):
    """Crea archivo temporal seguro"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"emotion_temp_{timestamp}_{uploaded_file.name}"
        temp_path = os.path.join(TEMP_DIR, temp_filename)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return temp_path
    except Exception as e:
        st.error(f"Error al crear archivo temporal: {str(e)}")
        return None


def cleanup_temp_files():
    """Limpia archivos temporales antiguos"""
    try:
        current_time = datetime.now()
        for filename in os.listdir(TEMP_DIR):
            if filename.startswith("emotion_temp_"):
                file_path = os.path.join(TEMP_DIR, filename)
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if current_time - file_time > timedelta(hours=RETENTION_HOURS):
                    os.remove(file_path)
    except Exception as e:
        pass  # Silenciar errores de limpieza


def display_results(image, results):
    """Muestra los resultados de detección"""
    if not results:
        st.warning("No se detectaron rostros en la imagen")
        return

    # Crear imagen con anotaciones
    annotated_image = np.array(image.copy())

    for i, result in enumerate(results):
        x, y, w, h = result['bbox']
        emotion_idx = result['emotion']
        confidence = result['confidence']

        emotion_data = EMOTION_LABELS[emotion_idx]

        # Dibujar rectángulo
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Agregar etiqueta
        label = f"{emotion_data['name']} ({confidence:.1%})"
        cv2.putText(annotated_image, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Mostrar imagen anotada
    st.image(annotated_image, caption="Emociones Detectadas",
             use_container_width=True)

    # Mostrar resultados detallados
    st.subheader("Emociones Detectadas:")

    for i, result in enumerate(results):
        emotion_idx = result['emotion']
        confidence = result['confidence']
        emotion_data = EMOTION_LABELS[emotion_idx]

        col1, col2, col3 = st.columns([1, 2, 3])

        with col1:
            st.markdown(f"<div style='font-size: 48px; text-align: center;'>{emotion_data['emoji']}</div>",
                        unsafe_allow_html=True)

        with col2:
            st.markdown(f"**{emotion_data['name']}**")
            st.markdown(f"Confianza: {confidence:.1%}")

        with col3:
            # Mostrar distribución de probabilidades
            for j, prob in enumerate(result['prediction']):
                emotion_name = EMOTION_LABELS[j]['name']
                st.progress(float(prob), text=f"{emotion_name}: {prob:.1%}")


def camera_capture():
    """Interfaz para captura desde cámara"""
    st.subheader("📷 Captura desde Cámara")

    camera_image = st.camera_input("Toma una foto para analizar emociones")

    if camera_image is not None:
        st.session_state["camera_image"] = camera_image
        return Image.open(camera_image)

    # Si ya había una foto tomada anteriormente (evita que se borre al cambiar de pestaña)
    if "camera_image" in st.session_state:
        return Image.open(st.session_state["camera_image"])

    return None


def file_upload():
    """Interfaz para subida de archivos"""
    st.subheader("📁 Seleccionar Archivo")

    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=ALLOWED_EXTENSIONS,
        help=f"Formatos permitidos: {', '.join(ALLOWED_EXTENSIONS)}. Tamaño máximo: {MAX_FILE_SIZE // (1024*1024)}MB"
    )

    if uploaded_file is not None:
        is_valid, message = validate_image_file(uploaded_file)

        if not is_valid:
            st.error(f"❌ {message}")
            return None

        st.success(f"✅ {message}")
        st.session_state["uploaded_image"] = uploaded_file
        return Image.open(uploaded_file)

    # Si ya había una imagen subida previamente
    if "uploaded_image" in st.session_state:
        return Image.open(st.session_state["uploaded_image"])

    return None


def main():
    """Función principal de la aplicación"""

    st.title("🎭 Detector de Emociones")
    st.markdown("---")

    with st.spinner("Cargando modelos..."):
        model = load_emotion_model()
        face_cascade = load_face_cascade()

    if model is None or face_cascade is None:
        st.error("No se pudieron cargar los modelos necesarios")
        return

    cleanup_temp_files()

    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        ### Emociones Detectadas:
        - 😊 **Happy** - Felicidad
        - 😢 **Sad** - Tristeza  
        - 😠 **Angry** - Enojo
        - 😨 **Fear** - Miedo
        - 😲 **Surprise** - Sorpresa
        - 🤢 **Disgust** - Asco
        - 😐 **Neutral** - Neutral
        
        ### Especificaciones:
        - **Formatos:** JPG, JPEG, PNG
        - **Tamaño máximo:** 5MB
        - **Eliminación automática:** 1 hora
        """)

    # Pestañas de entrada
    tab1, tab2 = st.tabs(["📁 Subir Archivo", "📷 Cámara"])

    # Solo una fuente de imagen activa a la vez
    image = None

    with tab1:
        if "uploaded_image" not in st.session_state:
            uploaded = file_upload()
            if uploaded:
                st.session_state["uploaded_image"] = uploaded
        if "uploaded_image" in st.session_state:
            image = st.session_state["uploaded_image"]

    with tab2:
        if "camera_image" not in st.session_state:
            captured = camera_capture()
            if captured:
                st.session_state["camera_image"] = captured
        if "camera_image" in st.session_state:
            image = st.session_state["camera_image"]

    # Procesamiento si hay imagen
    if image is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Imagen Original")
            st.image(image, caption="Imagen a analizar",
                     use_container_width=True)

        with col2:
            st.subheader("Análisis")
            if "emotion_results" not in st.session_state:
                with st.spinner("Analizando emociones..."):
                    results = detect_emotion(image, model, face_cascade)
                    st.session_state["emotion_results"] = results

                if st.session_state["emotion_results"]:
                    st.success(
                        f"✅ Se detectaron {len(st.session_state['emotion_results'])} rostro(s)")
                else:
                    st.warning("⚠️ No se detectaron rostros")

        # Mostrar resultados
        if st.session_state.get("emotion_results"):
            st.markdown("---")
            display_results(image, st.session_state["emotion_results"])

        # Botón de limpieza total
        if st.button("🗑️ Limpiar Resultados", type="secondary"):
            for key in ["camera_image", "uploaded_image", "emotion_results"]:
                st.session_state.pop(key, None)
            st.rerun()


if __name__ == "__main__":
    main()
