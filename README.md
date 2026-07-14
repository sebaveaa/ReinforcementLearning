# Entorno de Almacén Automatizado 5x5 (Gymnasium + Pygame)

Este repositorio contiene la primera fase de nuestro proyecto de **Aprendizaje por Refuerzo** para la clase de Inteligencia Artificial. Hemos diseñado la estructura modular del entorno personalizado heredando de `gymnasium.Env` y su renderizado visual con una estética retro pixel-art inspirada en Pac-Man.

---

## 📋 Resumen del Proyecto Actual

Hasta el momento se ha implementado:
1. **`warehouse_env.py` (Módulo del Entorno)**:
   * **Espacio de Acciones (Discrete 6)**: Mover arriba, abajo, izquierda, derecha, recoger paquete (pickup) y entregar paquete (drop-off).
   * **Espacio de Observaciones (MultiDiscrete)**: Representación de posiciones en la cuadrícula de 5x5 del robot, obstáculo, pickup, drop-off y el estado de la carga (0 o 1).
   * **Método `reset()`**: El obstáculo inicia en una posición fija en el centro `(2, 2)`. Las zonas de recogida (pickup) y entrega (drop-off) se generan aleatoriamente en cada inicio asegurando que sean distintas.
   * **Método `step(action)`**: Controla el movimiento del robot, mueve el obstáculo de forma estocástica (20% de probabilidad por dirección o quedarse quieto), calcula colisiones (término por choque con penalización de -100) y entregas exitosas (término por éxito con recompensa de +100).
   * **Renderizado Visual en Pygame**: Sprites pixel-art tiernos cargados dinámicamente en memoria (Pac-Man animado que rota, fantasma Blinky con ojos redondeados y cerezas como indicador de paquete).
2. **`requirements.txt`**: Archivo de dependencias del proyecto.
3. **`warehouse_demo.ipynb`**: Notebook de demostración para verificar el entorno en consola (ANSI) y gráficamente en Pygame con acciones aleatorias.

---

## 🚀 Guía de Instalación y Ejecución

### Requisito Recomendado: VS Code
Para una ejecución interactiva más cómoda, te recomendamos abrir este proyecto utilizando **VS Code** con las siguientes extensiones instaladas:
* **Python** (de Microsoft)
* **Jupyter** (de Microsoft)

---

### Paso 1: Instalar dependencias
Abre una terminal en la raíz de este proyecto y ejecuta:

```bash
pip install -r requirements.txt
```

> **Nota técnica:** Se está utilizando `pygame-ce` (Community Edition) en lugar del Pygame estándar debido a que ofrece soporte directo para versiones recientes de Python (como Python 3.12, 3.13 y 3.14) sin requerir compiladores adicionales en Windows.

---

### Paso 2: Ejecución en VS Code
1. Abre la carpeta del proyecto en **VS Code**.
2. Abre el archivo **`warehouse_demo.ipynb`**.
3. Selecciona tu Kernel de Python en la esquina superior derecha (asegúrate de que coincida con el entorno donde instalaste las dependencias).
4. Ejecuta las celdas secuencialmente:
   * La **primera celda** carga las librerías y el entorno.
   * La **segunda y tercera celda** prueban el entorno imprimiéndolo en formato de texto (ANSI) en la consola del notebook.
   * La **cuarta celda** abrirá una ventana de Pygame donde verás a Pac-Man y al Fantasma moviéndose aleatoriamente para verificar que las colisiones, recogida de cerezas y físicas del entorno funcionen correctamente.

---

## 🛠️ Estructura del Código

* **`warehouse_env.py`**: Código del entorno. Es totalmente modular. Cuando pasemos a la fase de entrenar los agentes (Q-Learning / Deep Q-Network), solo debemos importar la clase `WarehouseEnv` en nuestro script o notebook de entrenamiento.
* **`warehouse_demo.ipynb`**: Bucle principal con acciones aleatorias (`env.action_space.sample()`) para pruebas rápidas de control de calidad.
