import cv2
import numpy as np
import threading
import time
import os

class VideoProcessorCV:
    def __init__(self, video_path, estacionamientos, db):
        self.video = cv2.VideoCapture(video_path)
        self.estacionamientos = estacionamientos
        self.db = db
        self.estado_actual = [{"id": i, "ocupado": False, "reservado": False, "count": 0} 
                             for i in range(len(estacionamientos))]
        self.lock = threading.Lock()

    def generar_frames(self):
        while True:
            success, frame = self.video.read()

            if not success:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            reserved_spaces = self.db.get_active_reservations()

            img = frame.copy()
            imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 25, 16)
            imgMedian = cv2.medianBlur(imgTH, 5)
            kernel = np.ones((5,5), np.int8)
            imgDil = cv2.dilate(imgMedian, kernel)

            for i, puntos in enumerate(self.estacionamientos):
                pts = np.array(puntos, dtype=np.int32)

                mask = np.zeros(imgDil.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)

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
                    color = (255, 255, 0)
                elif ocupado:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)

                cv2.polylines(frame, [pts], True, color, 2)
                cv2.putText(frame, f"{i+1}", (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.03)

    def get_estado_espacios(self):
        reserved_spaces = self.db.get_active_reservations()
        for i, espacio in enumerate(self.estado_actual):
            espacio['reservado'] = (i + 1) in reserved_spaces
        return self.estado_actual