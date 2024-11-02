# Predicción de Enfermedades Cardiovasculares mediante Machine Learning y Procesamiento de Lenguaje Natural (NLP)

Este repositorio contiene el código y los recursos para un sistema de apoyo al diagnóstico clínico en cardiología. El proyecto está enfocado en predecir el riesgo de enfermedades cardiovasculares (ECV) mediante el uso de modelos de Machine Learning (ML) y procesamiento de lenguaje natural (NLP). Combinando un modelo de Random Forest para la predicción de riesgos con técnicas avanzadas de NLP basadas en la arquitectura Transformers, el sistema es capaz de extraer y analizar información relevante de textos médicos, proporcionando apoyo en la toma de decisiones clínicas.

## Características Principales

- **Predicción de Riesgo Cardiovascular**: 
  - Utiliza un modelo de Machine Learning, específicamente Random Forest, entrenado con un conjunto de datos clínicos de pacientes con factores de riesgo cardíaco. 
  - Variables clave incluyen la presión arterial, niveles de colesterol, resultados de ECG, y otros datos clínicos relevantes.

- **Asistente Biomédico NLP**: 
  - Implementación de un modelo Transformer (LLaMA) que permite responder preguntas médicas relacionadas con cardiología. 
  - Facilita el acceso a información crucial para el diagnóstico y análisis de riesgo, mejorando la toma de decisiones en la práctica clínica.

- **Recuperación de Información Médica**: 
  - Uso de embeddings y FAISS para indexar y recuperar documentos médicos relevantes en función de la consulta del usuario.
  - Proporciona contexto adicional al asistente NLP para mejorar la precisión y relevancia de las respuestas.

## Tecnologías Utilizadas

- **Machine Learning**: Random Forest para la predicción del riesgo de ECV.
- **Deep Learning**: Arquitectura Transformer (LLaMA) para procesamiento de texto y generación de respuestas.
- **Bibliotecas de Python**:
  - `pandas` para el manejo de datos.
  - `scikit-learn` para la implementación de algoritmos de Machine Learning.
  - `transformers` para la implementación de modelos NLP.
  - `FAISS` para la búsqueda de similitud de embeddings y recuperación de información.
  
## Objetivo

El objetivo de este proyecto es desarrollar una herramienta de apoyo clínico que mejore la precisión y rapidez en el diagnóstico de enfermedades cardíacas. La implementación permite realizar predicciones personalizadas sobre el riesgo de ECV y ofrecer respuestas médicas precisas a través de un asistente biomédico, contribuyendo a una atención médica más eficaz y personalizada.

## Estructura del Proyecto

- `app.py`: Archivo principal de la aplicación que ejecuta el sistema de predicción y NLP.
- `codigo_TFM.py`: Contiene las funciones y métodos para el modelo de predicción y procesamiento de datos.
- `home.html`: Página de inicio de la aplicación web que permite la interacción con el sistema.
- `README.md`: Documentación del proyecto.

## Instalación

### Prerrequisitos

- Python 3.8 o superior
- Git
- pip

### Clonación del Repositorio

Para clonar el repositorio, ejecuta los siguientes comandos en tu terminal:

```bash
git clone https://github.com/carpasgis2/TFM.git
cd TFNM



