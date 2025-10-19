import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# ----------------- CONFIGURACIÓN DE PÁGINA -----------------
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# ----------------- ESTILOS -----------------
st.markdown("""
    <style>
    .title {text-align: center; font-size: 40px; font-weight: bold; color: #2c3e50;}
    .subtitle {text-align: center; font-size: 20px; color: #7f8c8d; margin-bottom: 30px;}
    .footer {text-align: center; font-size: 13px; color: #95a5a6; margin-top: 40px;}
    .stDataFrame {border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# ----------------- FUNCIONES -----------------
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception:
                st.warning(f"Intentando método alternativo de carga...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Instalar versión compatible de PyTorch y YOLOv5:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. Verificar que el archivo del modelo esté en la ubicación correcta.
        3. Si persiste, intenta descargarlo desde torch hub.
        """)
        return None

# ----------------- ENCABEZADO -----------------
st.markdown("<p class='title'>🔍 Detección de Objetos con YOLOv5</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Captura una imagen con tu cámara y detecta objetos en tiempo real utilizando inteligencia artificial</p>", unsafe_allow_html=True)

# ----------------- CARGA DE MODELO -----------------
with st.spinner("⚙️ Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

# ----------------- APLICACIÓN -----------------
if model:
    st.sidebar.title("⚙️ Parámetros de Detección")
    with st.sidebar:
        st.subheader("Ajustes principales")
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")
        
        st.subheader("Opciones avanzadas")
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("⚠️ Algunas opciones avanzadas no están disponibles.")

    # Contenedor principal
    main_container = st.container()

    with main_container:
        picture = st.camera_input("📸 Capturar imagen", key="camera")

        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            with st.spinner("🔎 Detectando objetos..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()
            
            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 Imagen procesada")
                    results.render()
                    st.image(cv2_img, channels='BGR', use_container_width=True)
                
                with col2:
                    st.subheader("📊 Resultados de la detección")
                    label_names = model.names
                    category_count = {}
                    
                    for category in categories:
                        category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        if category_idx in category_count:
                            category_count[category_idx] += 1
                        else:
                            category_count[category_idx] = 1
                    
                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                        data.append({
                            "Categoría": label,
                            "Cantidad": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })
                    
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    else:
                        st.info("No se detectaron objetos con los parámetros actuales.")
                        st.caption("💡 Intenta reducir el umbral de confianza en la barra lateral.")
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
                st.stop()
else:
    st.error("❌ No se pudo cargar el modelo. Verifica dependencias e inténtalo de nuevo.")
    st.stop()

# ----------------- PIE DE PÁGINA -----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>🚀 Aplicación desarrollada con Streamlit + YOLOv5 + PyTorch | Detección de objetos en tiempo real</p>", unsafe_allow_html=True)

