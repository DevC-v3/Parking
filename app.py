from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import numpy as np
import threading
import time
from database import db
from auth import auth_bp, login_required
from reservations import reservations_bp

app = Flask(__name__)
app.secret_key = 'parking-intelligence-secret-key-2025'

app.register_blueprint(auth_bp)
app.register_blueprint(reservations_bp)

# Cargar polígonos de espacios en formato:
# [ [(x1,y1),(x2,y2),(x3,y3),(x4,y4)], ... ]
with open('espacios.pkl', 'rb') as file:
    estacionamientos = pickle.load(file)

estado_espacios = [
    {"id": i, "ocupado": False, "reservado": False, "count": 0}
    for i in range(len(estacionamientos))
]

class VideoProcessor:
    def __init__(self):
        self.video = cv2.VideoCapture('video.mp4')
        self.estado_actual = estado_espacios.copy()

    def generar_frames(self):
        while True:
            success, frame = self.video.read()

            if not success:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            reserved_spaces = db.get_active_reservations()

            img = frame.copy()
            imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 25, 16)
            imgMedian = cv2.medianBlur(imgTH, 5)
            kernel = np.ones((5,5), np.int8)
            imgDil = cv2.dilate(imgMedian, kernel)

            for i, puntos in enumerate(estacionamientos):
                pts = np.array(puntos, dtype=np.int32)

                # Crear una máscara poligonal del tamaño del frame
                mask = np.zeros(imgDil.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)

                # Contar pixeles solo dentro del polígono
                espacio = cv2.bitwise_and(imgDil, mask)
                count = cv2.countNonZero(espacio)

                ocupado = count >= 900
                reservado = (i + 1) in reserved_spaces

                self.estado_actual[i] = {
                    "id": i,
                    "ocupado": ocupado,
                    "reservado": reservado,
                    "count": count
                }

                if reservado:
                    color = (255, 255, 0)  # Amarillo (reservado)
                elif ocupado:
                    color = (255, 0, 0)    # Rojo (ocupado)
                else:
                    color = (0, 255, 0)    # Verde (libre)

                # Dibujar polígono
                cv2.polylines(frame, [pts], True, color, 2)

                # Etiqueta del número del espacio (se usa el primer punto para colocar texto)
                cv2.putText(frame, f"{i+1}", (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.03)

    def get_estado_espacios(self):
        reserved_spaces = db.get_active_reservations()
        for i, espacio in enumerate(self.estado_actual):
            espacio['reservado'] = (i + 1) in reserved_spaces
        return self.estado_actual

video_processor = VideoProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mapa')
@login_required
def mapa():
    return render_template('mapa.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_processor.generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/estado_espacios')
def get_estado_espacios():
    return jsonify(video_processor.get_estado_espacios())

@app.route('/reservas')
@login_required
def reservas():
    return render_template('reservas.html')

if __name__ == '__main__':
    print("📍 Iniciando Parking Intelligence System...")
    print("🔗 http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)