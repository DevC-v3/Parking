import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, video_path, estacionamientos, db, model_size='nano'):
        self.video = cv2.VideoCapture(video_path)
        self.estacionamientos = estacionamientos
        self.db = db
        self.estado_actual = [{"id": i, "ocupado": False, "reservado": False, "count": 0} 
                             for i in range(len(estacionamientos))]
        self.lock = threading.Lock()
        
        # YOLO
        self.model = YOLO('yolov8n.pt')
        self.clases_vehiculos = [2, 3, 5, 7]  # coche, moto, autobús, camión
        
        # Cache
        self.ultimas_detecciones = []
        self.frame_count = 0
        
        print("✅ YOLO inicializado")

    def _detectar_vehiculos_simple(self, frame):
        """Detección simple y directa"""
        try:
            # Reducir tamaño para más velocidad
            frame_pequeno = cv2.resize(frame, (640, 360))
            
            # Detección básica
            results = self.model(frame_pequeno, conf=0.5, classes=self.clases_vehiculos, verbose=False)
            
            detecciones = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Escalar coordenadas al tamaño original
                        scale_x = frame.shape[1] / 640
                        scale_y = frame.shape[0] / 360
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y) 
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        detecciones.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(box.conf[0])
                        })
            
            return detecciones
            
        except Exception as e:
            print(f"Error en detección: {e}")
            return []

    def _esta_vehiculo_en_espacio(self, espacio_puntos, vehiculo_bbox):
        """Verifica si un vehículo está dentro de un espacio"""
        x1, y1, x2, y2 = vehiculo_bbox
        centro_x = (x1 + x2) // 2
        centro_y = (y1 + y2) // 2
        
        # Verificar si el centro del vehículo está dentro del polígono
        return cv2.pointPolygonTest(espacio_puntos, (centro_x, centro_y), False) >= 0

    def _determinar_ocupacion(self, detecciones):
        """Determina qué espacios están ocupados"""
        ocupacion = [False] * len(self.estacionamientos)
        
        for i, puntos in enumerate(self.estacionamientos):
            pts = np.array(puntos, dtype=np.int32)
            
            for det in detecciones:
                if self._esta_vehiculo_en_espacio(pts, det['bbox']):
                    ocupacion[i] = True
                    break
                    
        return ocupacion

    def generar_frames(self):
        """Generador de frames simplificado"""
        while True:
            success, frame = self.video.read()

            if not success:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Detección cada 5 frames para mejor performance
            if self.frame_count % 5 == 0:
                self.ultimas_detecciones = self._detectar_vehiculos_simple(frame)

            ocupacion_actual = self._determinar_ocupacion(self.ultimas_detecciones)
            reserved_spaces = self.db.get_active_reservations()

            # Actualizar estado
            with self.lock:
                for i in range(len(self.estacionamientos)):
                    self.estado_actual[i] = {
                        "id": i,
                        "ocupado": ocupacion_actual[i],
                        "reservado": (i + 1) in reserved_spaces,
                        "count": 0
                    }

            # Dibujar resultados
            for i, puntos in enumerate(self.estacionamientos):
                pts = np.array(puntos, dtype=np.int32)
                ocupado = ocupacion_actual[i]
                reservado = (i + 1) in reserved_spaces
                
                if reservado:
                    color = (255, 255, 0)  # Amarillo
                elif ocupado:
                    color = (0, 0, 255)    # Rojo  
                else:
                    color = (0, 255, 0)    # Verde
                
                cv2.polylines(frame, [pts], True, color, 2)
                cv2.putText(frame, f"{i+1}", (pts[0][0], pts[0][1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Codificar frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            self.frame_count += 1
            time.sleep(0.03)  # Mismo timing que el modelo 1

    def get_estado_espacios(self):
        """Retorna el estado actual de los espacios"""
        reserved_spaces = self.db.get_active_reservations()
        with self.lock:
            for i, espacio in enumerate(self.estado_actual):
                espacio['reservado'] = (i + 1) in reserved_spaces
            return self.estado_actual.copy()

    def __del__(self):
        """Liberar recursos"""
        if hasattr(self, 'video'):
            self.video.release()