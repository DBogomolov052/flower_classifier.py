import cv2
import numpy as np
import tensorflow as tf

# Загружаем обученную модель
model = tf.keras.models.load_model('flower_classification.h5')

# Задаем пороги для цветов
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

# Читаем изображение
img = cv2.imread("flowers.jpg")

# Конвертируем в HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Ищем цвета
mask_red = cv2.inRange(hsv_img, lower_red, upper_red)
mask_green = cv2.inRange(hsv_img, lower_green, upper_green)
