# Predicción de Enfermedades Cardiovasculares mediante Machine Learning y Procesamiento de Lenguaje Natural (NLP)

Este repositorio contiene el código y recursos para un sistema de apoyo al diagnóstico clínico en cardiología, enfocado en predecir el riesgo de enfermedades cardiovasculares (ECV) mediante modelos de Machine Learning (ML) y procesamiento de lenguaje natural (NLP). El sistema combina un modelo predictivo de Random Forest para evaluar el riesgo de enfermedad cardíaca con técnicas avanzadas de NLP basadas en la arquitectura Transformers para la extracción y análisis de información relevante en textos médicos.

## Características Principales

- **Predicción de Riesgo Cardiovascular**: Modelo de Machine Learning entrenado con un dataset clínico de pacientes con factores de riesgo cardíaco, utilizando variables clave como presión arterial, niveles de colesterol y resultados de ECG.
- **Asistente Biomédico NLP**: Implementación de un modelo Transformer que permite responder consultas médicas relacionadas con cardiología, mejorando el acceso a información crucial para el diagnóstico.
- **Recuperación de Información Médica**: Uso de embeddings y FAISS para la recuperación de documentos médicos relevantes basados en la consulta, integrando contexto adicional al asistente de NLP.

## Tecnologías

- **Machine Learning**: Random Forest para la predicción de ECV.
- **Deep Learning**: Arquitectura Transformer (LLaMA) para el procesamiento de texto y generación de respuestas.
- **Bibliotecas**: 
  - `pandas` para el manejo de datos
  - `scikit-learn` para algoritmos de ML
  - `transformers` para NLP
  - `FAISS` para la búsqueda de similitud de embeddings

## Objetivo

El objetivo de este proyecto es ofrecer una herramienta de apoyo clínico que mejore la precisión y rapidez en el diagnóstico de enfermedades cardíacas, proporcionando una asistencia médica más personalizada y eficaz.

## Instalación




1. Clona este repositorio:

   ```bash
   git clone https://github.com/carpasgis2/TFM.git
   cd TFM

