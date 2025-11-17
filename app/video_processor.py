import cv2
import numpy as np
import threading
import time
import os
from ultralytics import YOLO
import torch

class VideoProcessor:
    def __init__(self, video_path, estacionamientos, db, model_size='nano'):
        self.video = cv2.VideoCapture(video_path)
        self.estacionamientos = estacionamientos
        self.db = db
        self.estado_actual = [{"id": i, "ocupado": False, "reservado": False, "count": 0} 
                             for i in range(len(estacionamientos))]
        self.lock = threading.Lock()
        
        # Inicializar YOLO con optimizaciones
        self.model = self._inicializar_yolo(model_size)
        
        # Pre-calcular bounding boxes de los espacios
        self.espacio_bboxes = self._precalcular_bboxes_espacios()
        
        # Optimizaciones de rendimiento
        self.historial_detecciones = [[] for _ in range(len(estacionamientos))]
        self.max_historial = 3
        
        # Variables para cache de detección
        self.last_vehicles = []
        self.last_ocupacion = [False] * len(self.espacio_bboxes)
        self.detection_interval = 5
        self.frame_count = 0
        
        print(f"✅ YOLO {model_size} inicializado - Optimizado para velocidad")

    def _inicializar_yolo(self, model_size):
        """Inicializa YOLO con optimizaciones de rendimiento"""
        model_paths = {
            'nano': 'yolov8n.pt',
            'small': 'yolov8s.pt', 
        }
        
        model_path = model_paths.get(model_size, 'yolov8n.pt')
        model = YOLO(model_path)
        
        # Optimizaciones para CPU
        if not torch.cuda.is_available():
            torch.set_num_threads(1)
        
        return model

    def _precalcular_bboxes_espacios(self):
        """Pre-calcula los bounding boxes de cada espacio"""
        bboxes = []
        for puntos in self.estacionamientos:
            pts = np.array(puntos, dtype=np.int32)
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            bbox = {
                'x1': min(x_coords),
                'y1': min(y_coords), 
                'x2': max(x_coords),
                'y2': max(y_coords),
                'polygon': pts
            }
            bboxes.append(bbox)
        return bboxes

    def _calcular_iou(self, bbox1, bbox2):
        """Calcula Intersection over Union optimizado"""
        x1 = max(bbox1['x1'], bbox2[0])
        y1 = max(bbox1['y1'], bbox2[1])
        x2 = min(bbox1['x2'], bbox2[2])
        y2 = min(bbox1['y2'], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def _detectar_vehiculos_optimizado(self, frame):
        """Detección de vehículos optimizada para velocidad"""
        try:
            # Reducir tamaño de imagen para procesamiento más rápido
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame
            
            # Detección más rápida con parámetros optimizados
            results = self.model(
                frame_resized, 
                conf=0.5,
                classes=[2, 3, 5, 7],
                verbose=False,
                imgsz=320,
                half=False,
            )
            
            vehicles = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Escalar coordenadas de vuelta si se redimensionó
                        if width > 640:
                            x1 = int(x1 / scale)
                            y1 = int(y1 / scale)
                            x2 = int(x2 / scale)
                            y2 = int(y2 / scale)
                        
                        vehicles.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence
                        })
            
            return vehicles
            
        except Exception as e:
            print(f"❌ Error en detección YOLO: {e}")
            return []

    def _determinar_ocupacion_rapida(self, vehicles):
        """Determina ocupación optimizada para velocidad"""
        ocupacion_actual = self.last_ocupacion.copy()
        
        for i, espacio_bbox in enumerate(self.espacio_bboxes):
            vehiculo_detectado = False
            
            for vehicle in vehicles:
                iou = self._calcular_iou(espacio_bbox, vehicle['bbox'])
                
                if iou > 0.25:
                    vehiculo_detectado = True
                    break
            
            # Actualizar historial simplificado
            self.historial_detecciones[i].append(vehiculo_detectado)
            if len(self.historial_detecciones[i]) > self.max_historial:
                self.historial_detecciones[i].pop(0)
            
            # Decisión más rápida
            if len(self.historial_detecciones[i]) > 0:
                positivos = sum(self.historial_detecciones[i])
                ocupacion_actual[i] = positivos > (len(self.historial_detecciones[i]) // 2)
            
        return ocupacion_actual

    def _dibujar_resultados_rapido(self, frame, ocupacion_espacios, reserved_spaces):
        """Dibuja resultados optimizado para velocidad"""
        
        for i, espacio_bbox in enumerate(self.espacio_bboxes):
            pts = espacio_bbox['polygon']
            ocupado = ocupacion_espacios[i]
            reservado = (i + 1) in reserved_spaces
            
            # Determinar color según estado
            if reservado:
                color = (255, 255, 0)  # Amarillo (reservado)
            elif ocupado:
                color = (0, 0, 255)    # Rojo (ocupado)
            else:
                color = (0, 255, 0)    # Verde (libre)
            
            # Dibujar polígono
            cv2.polylines(frame, [pts], True, color, 2)
            
            # Solo el número del espacio
            cv2.putText(frame, f"{i+1}", (espacio_bbox['x1'], espacio_bbox['y1'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def generar_frames(self):
        """Genera frames optimizado para máxima velocidad"""
        
        while True:
            success, frame = self.video.read()

            if not success:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Detección cada N frames
            if self.frame_count % self.detection_interval == 0:
                vehicles = self._detectar_vehiculos_optimizado(frame)
                self.last_ocupacion = self._determinar_ocupacion_rapida(vehicles)
                reserved_spaces = self.db.get_active_reservations()
            
            # Actualizar estado actual
            with self.lock:
                for i in range(len(self.espacio_bboxes)):
                    self.estado_actual[i] = {
                        "id": i,
                        "ocupado": self.last_ocupacion[i],
                        "reservado": (i + 1) in reserved_spaces,
                        "count": 0
                    }

            # Dibujar resultados
            self._dibujar_resultados_rapido(frame, self.last_ocupacion, reserved_spaces)

            # Codificar frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            self.frame_count += 1
            time.sleep(0.02)

    def get_estado_espacios(self):
        """Retorna el estado actual de los espacios"""
        reserved_spaces = self.db.get_active_reservations()
        with self.lock:
            for i, espacio in enumerate(self.estado_actual):
                espacio['reservado'] = (i + 1) in reserved_spaces
            return self.estado_actual.copy()

    def __del__(self):
        """Liberar recursos cuando se destruye el objeto"""
        if hasattr(self, 'video'):
            self.video.release()