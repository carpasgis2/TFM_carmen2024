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



## Instalación

### Prerrequisitos

- Python 3.8 o superior
- Git
- pip

### Clonación del Repositorio

```bash
git clone https://github.com/carpasgis2/TFM.git
cd TFM
Instalación de Dependencias
Instala las dependencias del proyecto ejecutando:

bash
Copiar código
pip install -r requirements.txt
Uso
Entrenamiento del Modelo de Machine Learning
Asegúrate de que el conjunto de datos esté en la carpeta data/.
Ejecuta el script de entrenamiento train_model.py para entrenar el modelo de predicción de riesgo cardiovascular.
bash
Copiar código
python scripts/train_model.py
Uso del Asistente Biomédico NLP
Ejecuta el asistente de NLP con el script run_nlp_assistant.py, proporcionando una consulta médica relacionada con cardiología.
bash
Copiar código
python scripts/run_nlp_assistant.py
Este asistente NLP procesará la consulta, recuperará información relevante y generará respuestas en base al contexto.
Recuperación de Información Médica
El sistema de recuperación utiliza FAISS para realizar búsquedas eficientes en la base de datos de documentos médicos. Estos fragmentos de texto se combinan con el asistente NLP para generar respuestas contextualizadas.

Ejemplos de Uso
bash
Copiar código
# Ejecutar el modelo de predicción de riesgo cardiovascular
python scripts/predict_risk.py --data data/sample_patient_data.csv

# Consultar al asistente biomédico sobre tratamiento para un tipo específico de dolor torácico
python scripts/run_nlp_assistant.py --query "¿Cuál es el tratamiento recomendado para la angina típica?"
Evaluación
El rendimiento del modelo de predicción se evalúa en términos de precisión, sensibilidad y especificidad. Los resultados incluyen gráficas y tablas generadas en results/ que muestran el rendimiento en distintos conjuntos de prueba.

Contribuciones
Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para sugerir mejoras o resolver problemas. Asegúrate de seguir las buenas prácticas de código y documentar cualquier cambio en el README.md cuando sea relevante.

Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.


