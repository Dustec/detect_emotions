# 🎭 Detector de Emociones - Streamlit App

Esta es una aplicación interactiva construida con [Streamlit](https://streamlit.io/) que permite detectar emociones faciales a partir de una imagen subida o una fotografía capturada con la cámara del dispositivo.

## 🚀 ¿Qué hace esta app?

- 📤 Permite **subir una imagen** (JPG, JPEG, PNG) desde tu dispositivo.
- 📷 También puedes **tomar una foto** con la cámara en tiempo real.
- 🧠 Utiliza un **modelo de deep learning preentrenado** para detectar emociones en los rostros presentes.
- 😊 Muestra los resultados con:
  - Emoji de emoción detectada.
  - Nombre y nivel de confianza.
  - Barra de progreso con la probabilidad de cada emoción.
- 🧼 Tiene un botón para **limpiar resultados** y reiniciar el análisis fácilmente.

---

## 🧠 Modelo de detección de emociones

La app utiliza un modelo preentrenado sobre el dataset **FER2013**, basado en la arquitectura `Mini-XCEPTION`. Este modelo fue entrenado para clasificar rostros en 7 emociones básicas:

| Índice | Emoción   | Emoji  |
|--------|-----------|--------|
| 0      | Angry     | 😠     |
| 1      | Disgust   | 🤢     |
| 2      | Fear      | 😨     |
| 3      | Happy     | 😊     |
| 4      | Neutral   | 😐     |
| 5      | Sad       | 😢     |
| 6      | Surprise  | 😲     |

### 📥 Fuente del modelo

Se usa el archivo `emotion_model.h5` disponible públicamente (convertido del trabajo de oarriaga):

- Basado en: [face_classification](https://github.com/oarriaga/face_classification)
- Input shape esperado: `(64, 64, 1)`
- Output: vector de 7 probabilidades (softmax)

> 💡 Puedes reemplazar este modelo por otro `.h5` compatible si lo deseas, siempre que respete el input shape.

---

## 🧩 Estructura de funciones

### 🔍 `detect_emotion(image, model, face_cascade)`

- Detecta rostros con OpenCV y predice emociones en cada uno.
- Retorna una lista de resultados por rostro, incluyendo bounding box, emoción y confianza.

### ⚙️ `preprocess_face(face_img)`

- Recorta, convierte a escala de grises, redimensiona y normaliza el rostro a 64x64.
- Retorna el array listo para usar con el modelo.

### 🧠 `load_emotion_model()`

- Carga el modelo `.h5` de emociones desde disco.
- Compila manualmente para evitar errores con versiones antiguas.

### 😎 `load_face_cascade()`

- Carga el clasificador HaarCascade para detección facial frontal (`OpenCV`).

### 📤 `file_upload()`

- Interfaz para subir una imagen desde el disco.
- Valida tamaño y formato antes de aceptar.

### 📷 `camera_capture()`

- Interfaz para tomar una foto desde la cámara del navegador.
- Usa `st.camera_input()` y persiste la imagen en `session_state`.

### 🧼 `cleanup_temp_files()`

- Elimina archivos temporales que hayan sido creados por la app después de 1 hora.

### 🎨 `display_results(image, results)`

- Dibuja las caras detectadas en la imagen.
- Muestra la emoción detectada y la distribución de probabilidades.

---

## 📦 Requisitos

Instala las dependencias con:

```bash
pip install streamlit opencv-python tensorflow pillow numpy
```

## ▶️ Cómo ejecutar

1. Asegúrate de tener Python 3.7 o superior.
2. Clona este repositorio o descarga los archivos.
3. Coloca el archivo `emotion_model.h5` en el mismo directorio que `app.py`.
4. Instala las dependencias necesarias con:

```bash
streamlit run app.py
```

## ✏️ Notas adicionales

Puedes usar tanto imágenes estáticas como capturas de cámara en vivo.

El modelo preentrenado que se incluye está optimizado para caras frontales y condiciones de iluminación aceptables.

Si deseas cambiar el modelo, asegúrate de que:

- Tenga salida de 7 clases (una por emoción).

- Use softmax como capa final.

- Reciba imágenes con dimensiones (64, 64, 1), o ajusta el preprocesamiento.

## 🧠 Referencias

- [📚 FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [🧠 Mini-XCEPTION Model](https://github.com/oarriaga/face_classification)
- [📘 Documentación de Streamlit](https://docs.streamlit.io/)
- [📷 OpenCV Haar Cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)
