import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Datos (asegúrate de que sean arreglos de numpy)
X = np.array([1, 1.5, 2, 2.5, 3, 3.2, 3.5, 4, 4.2, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Definición del modelo
model = keras.Sequential([
    layers.Dense(1, input_shape=(1,), activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento automático [cite: 76]
model.fit(X, y, epochs=10)