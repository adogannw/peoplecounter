import cv2
import torch
import tkinter as tk
from tkinter import filedialog

# Resim seç ve insanları bulmak için aktar
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
image = cv2.imread(file_path)
height, width, _ = image.shape

# YOLOv5 yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.2  # Güven eşiğini ayarlayın

results = model(image[:, :, ::-1])  # BGR'dan RGB'ye dönüştür

# Nesneleri ayrıştır
person_count = 0
for *xyxy, conf, cls in results.xyxy[0].tolist():
    if int(cls) == 0:  # 0: insanları temsil eden yolov5 sınıfı
        person_count += 1
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"insan: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# İnsan sayısını yaz
cv2.putText(image, f"Person count: {person_count}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Sonucu göster
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.resizeWindow("Image", (int(width/2), int(height/2)))
cv2.waitKey(0)
cv2.destroyAllWindows()
