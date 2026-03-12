import numpy as np

# Datos extendidos (25 datos)
horas_estudio = [1, 1.5, 2, 2.5, 3, 3.2, 3.5, 4, 4.2, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]
resultado_real = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Inicialización de parámetros según tu PDF [cite: 42, 43, 44]
peso = 0.1  # Valor inicial sugerido
bias = 0
alpha = 0.1 

# Entrenamiento por épocas [cite: 69]
for epoch in range(10):
    for i in range(len(horas_estudio)):
        x = horas_estudio[i]
        y_real = resultado_real[i]
        
        # Predicción simple (Regla de decisión) [cite: 54, 55]
        z = peso * x + bias
        y_pred = 1 if z >= 0 else 0
        
        # Cálculo de error y actualización [cite: 46, 47, 48]
        error = y_real - y_pred
        peso = peso + alpha * error * x
        bias = bias + alpha * error
        
    print(f"Epoch {epoch+1} finished. Peso: {peso:.2f}, Bias: {bias:.2f}") [cite: 72]