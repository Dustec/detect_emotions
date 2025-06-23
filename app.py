import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from datetime import datetime, timedelta
import threading
import pandas as pd
import hashlib
import io

# Intentar importar DeepFace (modelo de alta precisión)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="Detector de Emociones Avanzado",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de constantes
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']
TEMP_DIR = tempfile.gettempdir()
RETENTION_HOURS = 1

# Mapeo de emociones para DeepFace
DEEPFACE_EMOTIONS = {
    'angry': {"name": "Angry", "emoji": "😠", "color": "#FF6B6B"},
    'disgust': {"name": "Disgust", "emoji": "🤢", "color": "#A8E6CF"},
    'fear': {"name": "Fear", "emoji": "😨", "color": "#FFB347"},
    'happy': {"name": "Happy", "emoji": "😊", "color": "#FFD93D"},
    'neutral': {"name": "Neutral", "emoji": "😐", "color": "#C7C7C7"},
    'sad': {"name": "Sad", "emoji": "😢", "color": "#87CEEB"},
    'surprise': {"name": "Surprise", "emoji": "😲", "color": "#DDA0DD"}
}


@st.cache_resource
def load_emotion_model():
    """Carga el modelo apropiado según disponibilidad"""
    if DEEPFACE_AVAILABLE:
        placeholder = st.empty()
        with placeholder:
            st.success("✅ Usando DeepFace (Alta precisión)")
            time.sleep(3)
        placeholder.empty()
        return "deepface"
    else:
        st.error("❌ No hay modelos disponibles. Instale deepface o tensorflow.")
        return None


def get_image_hash(image):
    """Genera un hash único para la imagen para detectar cambios"""
    try:
        if isinstance(image, Image.Image):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
        else:
            img_bytes = image.getbuffer().tobytes()
        return hashlib.md5(img_bytes).hexdigest()
    except:
        return str(time.time())


def detect_emotion_deepface(image_path):
    """Detecta emociones usando DeepFace (alta precisión)"""
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        results = []
        iterable = result if isinstance(result, list) else [result]
        for face_data in iterable:
            emotion_scores = face_data['emotion']
            dominant_emotion = face_data['dominant_emotion']
            region = face_data.get('region', {})
            bbox = (
                region.get('x', 0),
                region.get('y', 0),
                region.get('w', 100),
                region.get('h', 100)
            )
            results.append({
                'bbox': bbox,
                'emotion': dominant_emotion,
                'confidence': emotion_scores[dominant_emotion] / 100.0,
                'all_emotions': emotion_scores,
                'method': 'deepface'
            })
        return results
    except Exception as e:
        st.error(f"Error en DeepFace: {str(e)}")
        return []


@st.cache_resource
def load_face_cascade():
    """Carga el clasificador de rostros de OpenCV"""
    try:
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        st.error(f"Error al cargar detector de rostros: {str(e)}")
        return None


def validate_image_file(uploaded_file):
    """Valida el archivo de imagen subido"""
    if uploaded_file is None:
        return False, "No se ha seleccionado ningún archivo"
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Formato no permitido. Use: {', '.join(ALLOWED_EXTENSIONS)}"
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"Archivo muy grande. Máximo: {MAX_FILE_SIZE // (1024*1024)}MB"
    return True, "Archivo válido"


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
        now = datetime.now()
        for fname in os.listdir(TEMP_DIR):
            if fname.startswith("emotion_temp_"):
                path = os.path.join(TEMP_DIR, fname)
                created = datetime.fromtimestamp(os.path.getctime(path))
                if now - created > timedelta(hours=RETENTION_HOURS):
                    os.remove(path)
    except Exception:
        pass


def display_results(image, results, use_expanders=True):
    """Muestra los resultados de detección con imagen reducida a 250x250"""
    if not results:
        st.warning("No se detectaron rostros en la imagen")
        return
    annotated = np.array(image.copy())
    for res in results:
        x, y, w, h = res['bbox']
        emo = res['emotion']
        conf = res['confidence']
        data = DEEPFACE_EMOTIONS.get(
            emo, {"name": emo.title(), "emoji": "🤖", "color": "#808080"})
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{data['name']} ({conf:.1%})"
        cv2.putText(annotated, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Mostrar imagen anotada en 250px de ancho
    st.image(annotated, caption="Emociones Detectadas", width=250)

    st.subheader("🎭 Análisis Detallado de Emociones")
    for i, res in enumerate(results):
        if use_expanders:
            with st.expander(f"👤 Rostro {i+1}", expanded=True):
                _render_emotion_detail(res, i)
        else:
            st.markdown(f"### 👤 Rostro {i+1}")
            _render_emotion_detail(res, i)


def _render_emotion_detail(res, i):
    """Subrutina para mostrar los detalles de emoción"""
    emo = res['emotion']
    conf = res['confidence']
    data = DEEPFACE_EMOTIONS.get(
        emo, {"name": emo.title(), "emoji": "🤖", "color": "#808080"})
    col1, col2, col3 = st.columns([1, 2, 4])
    with col1:
        st.markdown(
            f"<div style='font-size:48px;text-align:center'>{data['emoji']}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**{data['name']}**")
        st.markdown(f"Confianza: **{conf:.1%}**")
    with col3:
        st.markdown("**Distribución de Emociones:**")
        df = pd.DataFrame([{
            "Emoción": DEEPFACE_EMOTIONS.get(k, {"name": k.title()})["name"],
            "Probabilidad": v/100.0,
            "Emoji": DEEPFACE_EMOTIONS.get(k, {"emoji": "🤖"})["emoji"]
        } for k, v in res['all_emotions'].items()]).sort_values('Probabilidad', ascending=False)
        for _, row in df.iterrows():
            st.progress(
                row['Probabilidad'], text=f"{row['Emoji']} {row['Emoción']}: {row['Probabilidad']:.1%}")


def camera_capture():
    """Interfaz para captura desde cámara"""
    st.subheader("📷 Captura desde Cámara")
    cam = st.camera_input("Toma una foto para analizar emociones")
    if cam:
        st.session_state["camera_image"] = cam
        return Image.open(cam)
    if "camera_image" in st.session_state:
        return Image.open(st.session_state["camera_image"])
    return None


def file_upload():
    """Interfaz para subida de archivos"""
    st.subheader("📁 Seleccionar Archivo")
    up = st.file_uploader(
        "Selecciona una imagen",
        type=ALLOWED_EXTENSIONS,
        help=f"Formatos: {', '.join(ALLOWED_EXTENSIONS)}; Máx: {MAX_FILE_SIZE//(1024*1024)}MB"
    )
    if up:
        valid, msg = validate_image_file(up)
        if not valid:
            st.error(f"❌ {msg}")
            return None
        st.success(f"✅ {msg}")
        st.session_state["uploaded_image"] = up
        return Image.open(up)
    if "uploaded_image" in st.session_state:
        return Image.open(st.session_state["uploaded_image"])
    return None


def save_to_history(image, results):
    """Guarda la imagen procesada y sus resultados en el historial"""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    st.session_state["history"].insert(0, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_data": buf.getvalue(),
        "results": results
    })


def display_history():
    """Muestra el historial de análisis anteriores"""
    if not st.session_state.get("history"):
        st.info("📭 No hay imágenes en el historial aún.")
        return
    st.markdown("## 🕓 Historial de Análisis")
    for idx, entry in enumerate(st.session_state["history"]):
        with st.expander(f"🖼️ Análisis #{idx+1} - {entry['timestamp']}", expanded=False):
            display_results(Image.open(io.BytesIO(entry["image_data"])),
                            entry["results"], use_expanders=False)


def main():
    """Función principal de la aplicación"""
    st.title("🎭 Detector de Emociones Avanzado")
    st.markdown("---")

    with st.spinner("Cargando modelos..."):
        model = load_emotion_model()
        face_cascade = load_face_cascade() if model != "deepface" else None

    if model is None:
        st.error("❌ No se pudieron cargar los modelos necesarios")
        st.info(
            "💡 Para obtener la máxima precisión, instale DeepFace: `pip install deepface`")
        return

    cleanup_temp_files()

    # -- Consentimiento obligatorio antes de procesar imágenes --
    st.markdown("### 📝 Consentimiento de Uso de la Imagen")
    consentimiento = st.checkbox(
        "He leído y acepto que mi foto sea usada únicamente para el análisis de emociones",
        key="consentimiento"
    )
    if not consentimiento:
        st.warning("🔒 Debes otorgar tu consentimiento para continuar.")
        st.stop()
    # ----------------------------------------------------------

    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        ### 🎭 Emociones Detectadas:
        - 😊 **Happy**
        - 😢 **Sad**
        - 😠 **Angry**
        - 😨 **Fear**
        - 😲 **Surprise**
        - 🤢 **Disgust**
        - 😐 **Neutral**
        
        ### 📋 Especificaciones:
        - **Formatos:** JPG, JPEG, PNG
        - **Tamaño máximo:** 5MB
        - **Eliminación automática:** 1 hora
        """)
        st.markdown("---")
        display_history()
        with st.expander("ℹ️ Modelos Disponibles", expanded=False):
            st.markdown("""
            **🔬 DeepFace (Alta Precisión)**
            ```bash
            pip install deepface
            ```
            """)

    tabs = ["📁 Subir Archivo", "📷 Cámara"]
    sel = st.selectbox("Selecciona la fuente de entrada", tabs,
                       index=0 if st.session_state.get("active_tab") != "📷 Cámara" else 1)
    if st.session_state.get("active_tab") != sel:
        for k in ["camera_image", "uploaded_image", "emotion_results", "last_image_hash", "last_image_source"]:
            st.session_state.pop(k, None)
        st.session_state["active_tab"] = sel
        st.rerun()

    image = None
    temp_path = None
    src = None

    if sel == "📁 Subir Archivo":
        img = file_upload()
        if img:
            image = img
            src = "uploaded"
            st.session_state.pop("camera_image", None)
            if model == "deepface":
                temp_path = create_temp_file(
                    st.session_state["uploaded_image"])
    else:
        img = camera_capture()
        if img:
            image = img
            src = "camera"
            st.session_state.pop("uploaded_image", None)
            if model == "deepface":
                temp_path = create_temp_file(st.session_state["camera_image"])

    if image:
        cur_hash = get_image_hash(image)
        changed = False
        if (st.session_state.get("last_image_hash") != cur_hash or
                st.session_state.get("last_image_source") != src):
            changed = True
            st.session_state["last_image_hash"] = cur_hash
            st.session_state["last_image_source"] = src
            st.session_state.pop("emotion_results", None)

        st.subheader("🔍 Análisis")
        if "emotion_results" not in st.session_state or changed:
            with st.spinner("🧠 Analizando emociones..."):
                if model == "deepface" and temp_path:
                    res = detect_emotion_deepface(temp_path)
                else:
                    res = []
                st.session_state["emotion_results"] = res

        if st.session_state["emotion_results"]:
            st.success(
                f"✅ {len(st.session_state['emotion_results'])} rostro(s) detectado(s)")
        else:
            st.warning("⚠️ No se detectaron rostros")

        if st.session_state["emotion_results"]:
            st.markdown("---")
            display_results(image, st.session_state["emotion_results"])
            if changed:
                save_to_history(image, st.session_state["emotion_results"])

        if st.button("🗑️ Limpiar Resultados", type="secondary"):
            for k in ["camera_image", "uploaded_image", "emotion_results", "last_image_hash", "last_image_source"]:
                st.session_state.pop(k, None)
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            st.rerun()


if __name__ == "__main__":
    main()
