from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import numpy as np
import threading
import time
from database import db
from auth import auth_bp, login_required
from reservations import reservations_bp

# Configuración de Flask
app = Flask(__name__)
app.secret_key = 'parking-intelligence-secret-key-2025'  # Cambiar en producción

# Registrar Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(reservations_bp)

# Cargar configuración de espacios
with open('espacios.pkl', 'rb') as file:
    estacionamientos = pickle.load(file)

# Estado inicial de los espacios
estado_espacios = [
    {"id": i, "ocupado": False, "reservado": False, "count": 0} 
    for i in range(len(estacionamientos))
]

class VideoProcessor:
    """
    Procesador de video que detecta espacios ocupados y respeta reservas
    """
    
    def __init__(self):
        self.video = cv2.VideoCapture('video.mp4')
        self.estado_actual = estado_espacios.copy()
        
    def generar_frames(self):
        """Generar frames para streaming con detección y reservas"""
        while True:
            success, frame = self.video.read()
            
            if not success:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Obtener espacios reservados de la base de datos
            reserved_spaces = db.get_active_reservations()
            
            # Procesamiento de imagen para detección
            img = frame.copy()
            imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 25, 16)
            imgMedian = cv2.medianBlur(imgTH, 5)
            kernel = np.ones((5,5), np.int8)
            imgDil = cv2.dilate(imgMedian, kernel)
            
            # Analizar cada espacio
            for i, (x, y, w, h) in enumerate(estacionamientos):
                espacio = imgDil[y:y+h, x:x+w]
                count = cv2.countNonZero(espacio)
                ocupado = count >= 900
                reservado = (i + 1) in reserved_spaces
                
                # Actualizar estado
                self.estado_actual[i] = {
                    "id": i,
                    "ocupado": ocupado,
                    "reservado": reservado,
                    "count": count
                }
                
                # Determinar color según estado
                if reservado:
                    color = (255, 255, 0)  # Amarillo para reservado
                elif ocupado:
                    color = (255, 0, 0)    # Rojo para ocupado
                else:
                    color = (0, 255, 0)     # Verde para libre
                
                # Dibujar en el frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Preparar frame para streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)
    
    def get_estado_espacios(self):
        """Obtener estado actual actualizado con reservas"""
        reserved_spaces = db.get_active_reservations()
        for i, espacio in enumerate(self.estado_actual):
            espacio['reservado'] = (i + 1) in reserved_spaces
        return self.estado_actual

# Inicializar procesador de video
video_processor = VideoProcessor()

# ==================== RUTAS PRINCIPALES ====================

@app.route('/')
def index():
    """Página de inicio"""
    return render_template('index.html')

@app.route('/mapa')
@login_required
def mapa():
    """Página del mapa (requiere autenticación)"""
    return render_template('mapa.html')

@app.route('/video_feed')
def video_feed():
    """Streaming de video en tiempo real"""
    return Response(video_processor.generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/estado_espacios')
def get_estado_espacios():
    """API para estado de espacios (incluye reservas)"""
    return jsonify(video_processor.get_estado_espacios())

@app.route('/reservas')
@login_required
def reservas():
    """Página dedicada para hacer reservas"""
    return render_template('reservas.html')

# ==================== INICIO DE LA APLICACIÓN ====================

if __name__ == '__main__':
    print("📍 Iniciando Parking Intelligence System...")
    print("🔗 http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)