from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import numpy as np
import threading
import time

# Inicializar la aplicación Flask (servidor web)
app = Flask(__name__)

# CARGAR CONFIGURACIÓN DE ESPACIOS DE ESTACIONAMIENTO
# El archivo espacios.pkl contiene las coordenadas de cada espacio (x, y, ancho, alto)
# Esto se define previamente usando un script de calibración
with open('espacios.pkl', 'rb') as file:
    estacionamientos = pickle.load(file)

# Cada espacio tiene: id, estado de ocupación y contador de píxeles
estado_espacios = [{"id": i, "ocupado": False, "count": 0} for i in range(len(estacionamientos))]

# CLASE PRINCIPAL PARA PROCESAR EL VIDEO Y DETECTAR AUTOS
class VideoProcessor:
    
    def __init__(self):
        # Abrir el archivo de video para procesamiento
        self.video = cv2.VideoCapture('video.mp4')
        # Copiar el estado inicial de los espacios
        self.estado_actual = estado_espacios.copy()
        
    def generar_frames(self):
        
        # GENERADOR DE FRAMES PARA STREAMING EN TIEMPO REAL
        while True:
            # Leer el siguiente frame del video
            success, frame = self.video.read()
            
            # Si no se puede leer el frame (video terminado), reiniciar
            if not success:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Volver al inicio del video
                continue
            
            # PROCESAMIENTO DE IMAGEN PARA DETECCIÓN
            
            # Crear copia del frame original
            img = frame.copy()
            
            # 1. Convertir a escala de grises
            imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Aplicar threshold adaptativo para resaltar diferencias
            imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 25, 16)
            
            # 3. Aplicar filtro mediano para reducir ruido
            imgMedian = cv2.medianBlur(imgTH, 5)
            
            # 4. Crear kernel para operaciones morfológicas
            kernel = np.ones((5,5), np.int8)
            
            # 5. Dilatar las áreas para unir regiones cercanas
            imgDil = cv2.dilate(imgMedian, kernel)
            
            # ANÁLISIS DE CADA ESPACIO DE ESTACIONAMIENTO
            
            # Recorrer cada espacio definido en la configuración
            for i, (x, y, w, h) in enumerate(estacionamientos):
                # Extraer la región del espacio actual
                espacio = imgDil[y:y+h, x:x+w]
                
                # Contar píxeles blancos (área ocupada)
                count = cv2.countNonZero(espacio)
                
                # Determinar si está ocupado (threshold = 900 píxeles)
                ocupado = count >= 900
                
                # Actualizar estado en memoria
                self.estado_actual[i] = {
                    "id": i,
                    "ocupado": ocupado,
                    "count": count
                }
                
                # DIBUJAR EN EL FRAME PARA VISUALIZACIÓN
                # Verde = Libre, Rojo = Ocupado
                color = (0, 255, 0) if not ocupado else (255, 0, 0)
                
                # Dibujar rectángulo alrededor del espacio
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Agregar número de espacio
                cv2.putText(frame, f"{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # PREPARAR FRAME PARA STREAMING WEB
            
            # Codificar frame como JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Formato necesario para streaming MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Pequeña pausa para controlar FPS
            time.sleep(0.03)
    
    def get_estado_espacios(self):
        
        # OBTENER ESTADO ACTUAL DE TODOS LOS ESPACIOS
        # Usado por la interfaz web para actualizar el mapa
        
        return self.estado_actual

# INICIALIZACIÓN DEL SISTEMA

# Crear instancia del procesador de video
video_processor = VideoProcessor()

# RUTAS DE LA APLICACIÓN WEB
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mapa')
def mapa():
    return render_template('mapa.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_processor.generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/estado_espacios')
def get_estado_espacios():
    
    # API DE ESTADO - Endpoint JSON que devuelve el estado actual de todos los espacios
    return jsonify(video_processor.get_estado_espacios())

# INICIO DE LA APLICACIÓN
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)