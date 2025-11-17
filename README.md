# PassGAN -- Generación de contraseñas con Redes Generativas Adversarias

Este proyecto implementa una versión simplificada de **PassGAN**, un
modelo basado en Redes Generativas Adversarias (GANs) cuyo objetivo es
aprender patrones de contraseñas y generar nuevas contraseñas
plausibles. Además, permite evaluar su eficacia y analizar los riesgos
asociados a este tipo de modelos en el ámbito de la ciberseguridad.

------------------------------------------------------------------------

## 📂 Estructura del proyecto

El código está organizado en tres archivos principales:

### `utils.py`

Incluye funciones auxiliares: - Generación de un **dataset sintético**
de contraseñas. - Codificación y decodificación de contraseñas a
tensores. - Generador aleatorio de contraseñas (baseline de
comparación). - Cálculo de métricas: - **Entropía**. - **Hit-rate**.

### `passgan.py`

Contiene la implementación de la GAN: - **Generator**: genera secuencias
de caracteres que simulan contraseñas. - **Discriminator**: clasifica
secuencias como reales o generadas. - Función de **entrenamiento** de la
GAN. - Función para **generar contraseñas** una vez entrenado el modelo.

### `main.py`

Ejecuta el experimento completo: 1. Genera el dataset sintético. 2.
Entrena el modelo GAN. 3. Genera contraseñas nuevas. 4. Calcula métricas
de eficacia: - Entropía. - Hit-rate sobre el conjunto de test. 5. Genera
y guarda **gráficas**: - Distribución de longitudes. - Entropía
comparada. - Hit-rate GAN vs aleatorio.

Las figuras se guardan en la carpeta `figuras/`.
------------------------------------------------------------------------

## 📊 Resultados esperados

El programa genera: - Contraseñas sintéticas realistas. - Tres gráficas
comparativas: - **Distribución de longitudes** de contraseñas. -
**Entropía de caracteres** (real vs GAN vs aleatorio). - **Hit-rate**
sobre el conjunto de test.

También imprime un resumen con: - Número de contraseñas generadas. -
Métricas de entropía. - Eficacia de la GAN frente a un generador
aleatorio.

------------------------------------------------------------------------

