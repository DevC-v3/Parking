# app.py
from flask import Flask, render_template, Response, jsonify
import pickle
import os
from .video_processor import VideoProcessor
from .video_processorCV import VideoProcessorCV
from .database import db
from .auth import auth_bp, login_required, admin_required
from .reservations import reservations_bp

# Obtener el directorio base de la aplicación
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'parking-intelligence-secret-key-2025'

app.register_blueprint(auth_bp)
app.register_blueprint(reservations_bp)

with open(os.path.join(BASE_DIR, 'espacios.pkl'), 'rb') as file:
    estacionamientos = pickle.load(file)

# --------SELECCIONAR MODELO------------

video_processor = VideoProcessor(
    video_path=os.path.join(BASE_DIR, 'video.mp4'),
    estacionamientos=estacionamientos,
    db=db,
    model_size='small'
)

# video_processorCV = VideoProcessorCV(
#     video_path=os.path.join(BASE_DIR, 'video.mp4'),
#     estacionamientos=estacionamientos,
#     db=db,
# )

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

@app.route('/admin')
@admin_required
def admin():
    """Panel de administración - solo para usuarios con admin=True"""
    users = db.get_all_users()
    reservations = db.get_all_reservations()
    return render_template('admin.html', users=users, reservations=reservations)

@app.route('/api/admin/reservations')
@admin_required
def get_all_reservations():
    """Obtener todas las reservas (solo admin)"""
    reservations = db.get_all_reservations()
    return jsonify(reservations)

@app.route('/api/admin/users')
@admin_required
def get_all_users():
    """Obtener todos los usuarios (solo admin)"""
    users = db.get_all_users()
    return jsonify(users)