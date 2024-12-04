import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Cargar dataset Iris y entrenar un modelo de ejemplo
iris = load_iris()
X, y = iris.data, iris.target
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Función para simular la extracción de características de una imagen
def extraer_caracteristicas(imagen):
    """
    Simula la extracción de características desde una imagen.
    Aquí deberías implementar una red neuronal u otro modelo avanzado.
    """
    # Simular características como valores aleatorios del rango de los datos originales
    return np.random.uniform(low=4.0, high=8.0, size=(1, 4))

# Función para clasificar la imagen
def predecir_especie(imagen):
    caracteristicas = extraer_caracteristicas(imagen)
    prediccion = clf.predict(caracteristicas)
    return iris.target_names[prediccion[0]]

# Capturar video en tiempo real
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la imagen.")
        break

    # Voltear la imagen horizontalmente para crear el efecto espejo
    frame = cv2.flip(frame, 1)

    # Simular la detección de la flor: definimos una zona en la imagen donde "detectamos" la flor
    # (En un caso real, deberías usar técnicas de segmentación para detectar la flor)
    height, width, _ = frame.shape
    x, y, w, h = int(width * 0.3), int(height * 0.3), int(width * 0.4), int(height * 0.4)  # Definir una zona de "flor"

    # Dibujar un cuadro alrededor de la "flor" 
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 50, 150), 2)  # Rectángulo anaranjado

    # Simular una predicción para la "flor" detectada
    especie_predicha = predecir_especie(frame)

    # Colocar el texto en la imagen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Especie: {especie_predicha}", (x, y - 10), font, 1, (0, 65, 155), 2, cv2.LINE_AA)

    # Mostrar el cuadro capturado
    cv2.imshow("Cámara - Clasificador de Flores Iris", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
