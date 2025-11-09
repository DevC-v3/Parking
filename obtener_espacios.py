import cv2
import pickle

img = cv2.imread('estacionamiento.png')
clone = img.copy()

espacios = []          # Lista final: cada elemento = 4 puntos (un polígono)
puntos_actuales = []   # Puntos temporales del espacio que estás marcando

def click_event(event, x, y, flags, param):
    global puntos_actuales, espacios, img

    if event == cv2.EVENT_LBUTTONDOWN:
        # Guardar punto
        puntos_actuales.append((x, y))
        
        # Dibujar punto
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        # Si ya marcaste 4 puntos → formar el polígono
        if len(puntos_actuales) == 4:
            # Dibujar polígono en la imagen
            pts = cv2.polylines(img, [np.array(puntos_actuales, dtype=np.int32)], True, (255, 0, 0), 2)
            
            # Guardarlo en la lista final
            espacios.append(puntos_actuales.copy())

            # Limpiar lista temporal
            puntos_actuales = []

        cv2.imshow("Marcar espacios", img)

import numpy as np

cv2.imshow("Marcar espacios", img)
cv2.setMouseCallback("Marcar espacios", click_event)

print("INSTRUCCIONES:")
print("- Haz clic en 4 puntos para marcar un espacio (paralelogramo).")
print("- Cada 4 puntos se dibuja automáticamente.")
print("- Cuando termines todos los espacios, presiona 'S' para guardar.")
print("- Presiona 'ESC' para salir sin guardar.\n")

while True:
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        print("Cancelado. No se guardaron cambios.")
        break

    if key == ord('s') or key == ord('S'):
        with open('espacios.pkl', 'wb') as file:
            pickle.dump(espacios, file)
        print("\n✅ Archivo 'espacios.pkl' guardado correctamente.")
        print(f"Total de espacios marcados: {len(espacios)}")
        break

cv2.destroyAllWindows()
