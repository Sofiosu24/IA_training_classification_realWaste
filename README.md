# Fase 1 — Preprocesado de Datos

Clasificación de residuos en imágenes mediante CNN — Dataset **RealWaste**

Esta carpeta contiene la primera fase del proyecto: la **generación/selección del set de datos** y el **preprocesado** previo al entrenamiento del modelo.

---

## 📑 Tabla de contenidos

- [Reconocimiento del uso de IA](#-reconocimiento-del-uso-de-ia)
- [1. Selección del Dataset](#1-selección-del-dataset)
- [2. División Train / Validation / Test](#2-división-train--validation--test)
- [3. Preprocesado de los Datos](#3-preprocesado-de-los-datos)
- [4. Data Augmentation](#4-data-augmentation)
- [5. Verificaciones y Visualizaciones](#5-verificaciones-y-visualizaciones)

---

## 🤖 Reconocimiento del uso de IA

Para el desarrollo de este trabajo se utilizó el asistente de IA **Claude (Anthropic)** como apoyo en el proceso de aprendizaje.

Específicamente, el asistente apoyó en:

1. La elección y justificación del split **70/15/15** frente a otras alternativas como 80/10/10.
2. La reorganización de la estructura de carpetas para hacerla compatible con `flow_from_directory` de Keras.
3. La propuesta de bloques de verificación adicionales no incluidos en los notebooks vistos en clase, como el conteo de imágenes en cada split (para verificar que la separación física en carpetas coincide con el plan teórico) y la visualización de una imagen de cada clase (para confirmar la integridad de los archivos, detectar posibles errores de etiquetado y conocer el tamaño real de las imágenes).
4. La redacción de este documento de justificación.

> ⚠️ **Importante:** todas las decisiones finales fueron revisadas, comprendidas y validadas personalmente. El código fue ejecutado y verificado en Google Colab, y las decisiones técnicas se contrastaron contra el material proporcionado por en clase.

---

## 1. Selección del Dataset

### 1.1 Por qué RealWaste

Inicialmente se contempló trabajar con un dataset de imágenes de microscopía de fluorescencia para segmentación de patógenos. Sin embargo, este dataset presentaba múltiples complicaciones:

- Las imágenes estaban en formato `.tif`, que requiere librerías especializadas (`tifffile`) para su manejo en Colab.
- El problema era de **segmentación** (etiqueta por píxel), no de **clasificación** (etiqueta por imagen).
- El dataset incluía estructura compleja (imágenes sintéticas vs reales, with/without fungi).

Por estas razones se optó por [**RealWaste**](https://www.mdpi.com/2078-2489/14/12/633), un dataset publicado por la Universidad de Wollongong (Australia) con **4,752 imágenes** clasificadas en **9 categorías** de residuos. Sus ventajas:

- Formato JPG estándar, soportado nativamente por TensorFlow/Keras.
- Tamaño uniforme de **524×524 píxeles**.
- Tarea clara de clasificación multi-clase.
- Tamaño manejable para entrenar en Colab.

### 1.2 Distribución de clases

| Clase | Imágenes | % |
|---|---:|---:|
| Plastic | 921 | 19.4% |
| Metal | 790 | 16.6% |
| Paper | 500 | 10.5% |
| Miscellaneous Trash | 495 | 10.4% |
| Cardboard | 461 | 9.7% |
| Vegetation | 436 | 9.2% |
| Glass | 420 | 8.8% |
| Food Organics | 411 | 8.6% |
| Textile Trash | 318 | 6.7% |
| **TOTAL** | **4,752** | **100%** |

Se observa un **desbalance importante**: Plastic tiene casi 3× más imágenes que Textile Trash (proporción 2.9:1). Se considerará en fases posteriores el uso de `class_weight` durante el entrenamiento.

---

## 2. División Train / Validation / Test

### 2.1 Por qué tres conjuntos en lugar de dos

Aunque el enunciado pedía solo train y test, se decidió usar **tres conjuntos**:

- **Train:** el modelo aprende patrones.
- **Validation:** ajustar hiperparámetros y monitorizar overfitting durante el entrenamiento.
- **Test:** "prueba final" — solo se usa al final para reportar la métrica honesta.

Si se usara el test set para ajustar hiperparámetros, el modelo terminaría "optimizado" para ese conjunto y la accuracy reportada no reflejaría su verdadero desempeño.

### 2.2 Por qué 70/15/15 y no 80/10/10

Para datasets en el rango 1,000–10,000 imágenes, **70/15/15 es preferible** por tres razones:

**Razón 1 — La diferencia en train es marginal.** Pasar de 3,326 a 3,801 imágenes (+14%) tiene impacto mínimo con transfer learning o data augmentation.

**Razón 2 — La diferencia en validation/test es estadísticamente significativa.**

| Split | Test | Margen de error (95% confianza) |
|---|---|---|
| 80/10/10 | 476 imágenes | ±4.5% |
| 70/15/15 | 713 imágenes | **±3.7%** |

**Razón 3 — La clase minoritaria sería demasiado pequeña con 80/10/10.** Textile Trash, con solo 318 imágenes, quedaría con:
- 80/10/10: 32 imágenes en validation/test (métricas muy ruidosas).
- 70/15/15: 48 imágenes en validation/test (50% más estable).

### 2.3 Por qué split estratificado

El 70/15/15 se respetó **dentro de cada clase**, no sobre el dataset completo. Esto garantiza que cada split mantenga la misma proporción relativa de clases que el dataset original.

### 2.4 Resultado final

| Clase | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Cardboard | 322 | 70 | 69 | 461 |
| FoodOrganics | 287 | 62 | 62 | 411 |
| Glass | 294 | 63 | 63 | 420 |
| Metal | 553 | 119 | 118 | 790 |
| MiscellaneousTrash | 346 | 75 | 74 | 495 |
| Paper | 350 | 75 | 75 | 500 |
| Plastic | 644 | 138 | 139 | 921 |
| TextileTrash | 222 | 48 | 48 | 318 |
| Vegetation | 305 | 65 | 66 | 436 |
| **TOTAL** | **3,323 (69.9%)** | **715 (15.0%)** | **714 (15.0%)** | **4,752** |

### 2.5 Estructura física de carpetas

Se reorganizó para ser compatible con `flow_from_directory` de Keras (primero el split, después las clases):

```
RealWaste/
├── train/
│   ├── Cardboard/        (322 imágenes)
│   ├── FoodOrganics/     (287 imágenes)
│   └── ... (9 clases)
├── validation/
│   └── (mismas 9 clases)
└── test/
    └── (mismas 9 clases)
```
---

## 3. Preprocesado de los Datos

### 3.1 Librerías utilizadas

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

- **`os`**: rutas, listar archivos, crear carpetas.
- **`numpy`**: manipulación de arreglos numéricos.
- **`matplotlib.pyplot`**: visualización de imágenes.
- **`tensorflow`**: framework de deep learning.
- **`ImageDataGenerator`**: clase central — escalamiento, augmentation, lectura desde carpetas, entrega en batches.

> Se usa `os.path.join()` en lugar de concatenar strings con `+` porque maneja separadores automáticamente según el SO, evita problemas con barras dobles, y es el estándar de la industria.

### 3.2 Escalamiento de píxeles

Las imágenes vienen con valores [0, 255]. Se aplicó normalización min-max:

```
pixel_escalado = pixel_original / 255
```

Esto lleva los valores al rango **[0, 1]**, necesario para:

- **Estabilidad del entrenamiento:** evita gradientes desproporcionados.
- **Coherencia entre features:** ninguna característica domina por escala.

Se aplica a **train, validation y test** porque el modelo debe ver el mismo rango.

### 3.3 Hiperparámetros principales

| Parámetro | Valor | Justificación |
|---|---|---|
| `target_size` | (150, 150) | ~12× más rápido que 524×524, sin pérdida significativa. Mismo valor que el profesor en Cats_and_Dogs. |
| `batch_size` | 20 | Gradientes estables, bajo uso de RAM. Mismo valor del profesor. |
| `class_mode` | `'categorical'` | 9 clases → vector one-hot. Implica `categorical_crossentropy` y `softmax` en la capa de salida. |

---

## 4. Data Augmentation

### 4.1 Qué es y por qué se aplica

Aplicación de transformaciones aleatorias a las imágenes durante el entrenamiento para generar variaciones artificiales. Cada época el modelo ve versiones diferentes, lo cual:

- **Reduce overfitting** (no se memorizan imágenes exactas).
- **Mejora generalización** (reconoce objetos en distintas orientaciones, posiciones, escalas).
- **Compensa el tamaño limitado del dataset.**

Las transformaciones se aplican al vuelo en RAM, sin guardar archivos.

### 4.2 Transformaciones aplicadas

```python
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
```

| Parámetro | Valor | Justificación para residuos |
|---|---|---|
| `rotation_range` | 40 | Los residuos no tienen orientación canónica |
| `width_shift_range` | 0.2 | El objeto no siempre está centrado |
| `height_shift_range` | 0.2 | Idem vertical |
| `shear_range` | 0.2 | Simula cámara ligeramente inclinada |
| `zoom_range` | 0.2 | Variabilidad de distancia de captura |
| `horizontal_flip` | True | La basura no tiene "lado correcto" |

**No se usó `vertical_flip`** porque las fotos tienen orientación natural impuesta por la gravedad. Voltear verticalmente generaría imágenes irrealistas.

### 4.3 Diferencia crítica: train vs validation/test

| Set | Augmentation | Rescale |
|---|:---:|:---:|
| Train | ✅ Sí | ✅ Sí |
| Validation | ❌ No | ✅ Sí |
| Test | ❌ No | ✅ Sí |

**Validation:** sin augmentation porque la métrica debe ser estable entre épocas. Si aplicáramos transformaciones, no sabríamos si una mejora viene del modelo o del azar.

**Test:** sin augmentation porque la métrica final debe reflejar el desempeño en imágenes reales, no en versiones artificiales.

**Rescale en los 3:** el modelo aprendió con valores [0, 1]; debe ver el mismo rango en evaluación.Sin embargo, el rescalamineto de test se hace al final, en la etapa de evaluación.

---

## 5. Verificaciones y Visualizaciones

### 5.1 Sanity check del split

Se verificó que el split físico coincidiera con el plan teórico (70/15/15). Resultado: ✅ confirmado (3,323 / 715 / 714).

### 5.2 Visualización de una imagen por clase

Sirvió para confirmar que:

- ✅ Las imágenes no están corruptas.
- ✅ No hay errores de etiquetado entre clases.
- ✅ Todas son 524×524 (tamaño uniforme).

### 5.3 Visualización del data augmentation

Se generaron 5 versiones de una misma imagen pasándola por el generador. Las transformaciones (rotación, desplazamiento, espejado, zoom) son visiblemente distintas en cada versión.

### 5.4 Guardado de 15 imágenes aumentadas

Mediante `save_to_dir`, se guardaron 15 imágenes aumentadas en `/augmented` como evidencia visual. La elección de 15:

- Suficientes para mostrar variedad sin saturar Drive.
- Material visual para reportes/presentaciones.

> Durante el entrenamiento real, el modelo procesa **miles** de imágenes aumentadas por época sin guardarlas (el `train_generator` no tiene `save_to_dir` activado).

---

## 📚 Referencias

- Single, S., Iranmanesh, S., & Raad, R. (2023). [*RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning*](https://www.mdpi.com/2078-2489/14/12/633). MDPI Information.
- Notebooks del curso: `Tensorflow_Mnist.ipynb`, `Cats_and_Dogs_Keras.ipynb`, `Data_augmentation.ipynb`.

---

*Última actualización: [23 de Mayo de 2026]*

