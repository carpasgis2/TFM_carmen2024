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
  ```
### License
**Copyright © 2024 Carmen Pascual**

**Todos los derechos reservados.**
 
Este software y el código fuente asociado son propiedad exclusiva de Carmen Pascual y no pueden ser utilizados, copiados, modificados, ni distribuidos sin el permiso explícito y por escrito de los titulares de los derechos de autor.

**Términos de Uso:**
 
1. **Uso Personal y No Comercial**: Solo está permitido el uso personal y no comercial de este software. No se permite ningún uso en aplicaciones comerciales ni en entornos de producción sin una licencia comercial específica otorgada por Carmen Pascual.
 
2. **Prohibición de Modificación y Distribución**: No está permitida la modificación, distribución, sublicencia o cualquier forma de redistribución del código o del software en su totalidad o en parte sin autorización previa por escrito.
 
3. **Limitación de Responsabilidad**: Este software se proporciona "tal cual", sin garantías de ningún tipo, ya sean explícitas o implícitas, incluyendo, entre otras, garantías de comerciabilidad, adecuación para un propósito particular, o no infracción. En ningún caso los autores serán responsables de cualquier reclamo, daño u otra responsabilidad que surja del uso o de la imposibilidad de usar el software.
 
4. **Uso de Datos y Privacidad**: Cualquier dato utilizado o procesado por este software debe cumplir con todas las leyes y regulaciones de privacidad vigentes. Carmen Pascual no se responsabiliza de la forma en que los usuarios gestionen los datos personales o clínicos procesados por este software.
 
5. **Restricciones en Ingeniería Inversa**: Queda prohibido desensamblar, descompilar, o aplicar ingeniería inversa a cualquier parte de este software.
 
Para obtener más información sobre los términos de la licencia o para adquirir una licencia comercial, por favor contacta a: [Tu dirección de contacto o la de tu organización]
 
### Keywords
- Cardiovascular diseases (CVD)
- Myocardial infarction
- Machine Learning (ML)
- Deep Learning (DL)
- Artificial intelligence in cardiology
- Clinical risk factors
- Clinical decision support
- Natural Language Processing (NLP)
- Transformers architecture
