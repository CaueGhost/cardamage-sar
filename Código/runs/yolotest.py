from ultralytics import YOLO
import os

modelo = YOLO(r"C:\CODIGOSAR\Código\runs\detect\yolov8_car_damage-2\weights\best.pt")

resultado = modelo.predict(
    source=r"C:\yolo_car_damage\images\val",
    save=True,
    conf=0.1
)
print("Predições salvas!")