from flask import Flask, render_template, request, jsonify, session
import dill as pickle
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import requests
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

app = Flask(__name__)
# Cargar el scaler y el modelo entrenado
scaler_filename = 'scaler.pkl'
model1_filename = 'modelo_entrenado.pkl'

# Verificar que scaler y modelo están cargados correctamente
try:
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)
    print(f"{scaler_filename} cargado correctamente.")
    print(f"Tipo de scaler: {type(scaler)}")

    with open(model1_filename, 'rb') as f:
        model1 = pickle.load(f)
    print(f"{model1_filename} cargado correctamente.")
    print(f"Tipo de modelo: {type(model1)}")
except Exception as e:
    print("Error al cargar scaler o modelo:")
    traceback.print_exc()


# Configuración del tokenizer y modelo de embeddings
model_name = "jinaai/jina-embeddings-v2-base-es"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_embeddings = AutoModel.from_pretrained(model_name, trust_remote_code=True)
device = torch.device("cpu")
model_embeddings.to(device)


# Definir la clase LangChain_Embeddings
class LangChain_Embeddings(Embeddings):
    def __init__(self, embedder: AutoModel):
        self.embedder = embedder
        super().__init__()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            embeddings = self.embedder(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> list[float]:
        inputs = tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True, max_length=200
        )
        with torch.no_grad():
            embedding = self.embedder(**inputs).last_hidden_state.mean(dim=1)
        return embedding.cpu().numpy().tolist()[0]

def tryload(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, AttributeError) as e:
        print(f"Error al cargar {file_name}: {e}")
        return None

def save(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    print(f"Archivo guardado correctamente en {file_name}")


def load_document_pypdf2(pdf_path):
    documents = []
    try:
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page_number in range(len(reader.pages)):
                page = reader.pages[page_number]
                content = page.extract_text()
                if content:
                    documents.append({
                        "page_content": content.strip(),
                        "metadata": {"page": page_number + 1}
                    })
    except Exception as e:
        print(f"No se pudo cargar el archivo PDF: {e}")
    return documents

# Función para dividir documentos en fragmentos más pequeños
def split_documents(documents):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for doc in documents:
        doc_obj = Document(
            page_content=doc["page_content"], metadata=doc["metadata"]
        )
        chunks.extend([
            Document(page_content=chunk, metadata=doc["metadata"])
            for chunk in splitter.split_text(doc_obj.page_content)
        ])
    return chunks

# Cargar o crear el índice FAISS
faiss_index_filename = 'faiss_indexML2.pkl'
if os.path.exists(faiss_index_filename):
    with open(faiss_index_filename, 'rb') as f:
        faiss_index = pickle.load(f)
else:
    pdf_path = "Manual_AMIR_12da_ed_Cardiologia_y_Cirugi.pdf"
    document = load_document_pypdf2(pdf_path)
    chunks = split_documents(document)
    faiss_index = FAISS.from_documents(chunks, LangChain_Embeddings(model_embeddings))
    save(faiss_index_filename, faiss_index)

# Configurar el buscador con FAISS
faiss_retriever = faiss_index.as_retriever(search_kwargs={'k': 10})

# Función para obtener documentos relevantes
def get_documents(query, retriever):
    unique_docs = retriever.get_relevant_documents(query)
    documents_with_pages = []
    context = ""
    for doc in unique_docs:
        page_info = doc.metadata.get('page', 'Página desconocida')
        content = f"[Página {page_info}]\n{doc.page_content}\n\n"
        context += content
        documents_with_pages.append({
            "page": page_info,
            "content": doc.page_content,
        })
    return context, documents_with_pages

# Clase para manejar el asistente biomédico
class BiomedicalAssistant:
    def __init__(self, base_url, retriever, api_key=None):
        self.base_url = base_url
        self.retriever = retriever
        self.api_key = api_key
        self.sys_message = """"
Eres un asistente especializado en enfermedades cardíacas. Tu tarea es proporcionar respuestas claras, objetivas y basadas exclusivamente en los datos médicos y el contenido del documento proporcionado. Debes ofrecer interpretaciones precisas para el diagnóstico y manejo de condiciones cardiovasculares, como infarto de miocardio, hipertensión, arritmias, entre otras.

Instrucciones de Respuesta:

Contenido Relevante: Responde únicamente con la información relevante del documento o de fuentes científicas cuando no haya datos en el documento.
Formato: Organiza las respuestas en párrafos bien estructurados. Usa negritas para conceptos clave y listas numeradas cuando sea necesario.
Referencias: Al final de la respuesta, incluye una sección titulada "Referencias:" en un párrafo separado. Enumera las referencias en este formato: "- [1] Documento proporcionado (Página X)." Solo usa referencias directas del documento y, si no hay, utiliza el nombre de la fuente científica, indicando el artículo o fuente específica.

Respuestas Específicas:

Si la consulta es "¿qué enfermedad tiene el paciente?", utiliza exclusivamente los datos proporcionados en "Datos del paciente".
Si la pregunta es "¿cuáles son los últimos avances...?", busca únicamente en artículos recientes de PubMed."

"""

    
    def send_request(self, prompt):
        headers = {
            'Content-Type': 'application/json',
        }
        
        # Add API key if necessary
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        payload = {
            'prompt': prompt,
            'system_message': self.sys_message
        }
        
        response = requests.post(f'{self.base_url}/chat', json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f'Error en la solicitud: {response.status_code} - {response.text}')
            
            
    def send_chat(self, messages):
        headers = {
            'Content-Type': 'application/json',
        }
        
        # Add API key if necessary
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        payload = {
            'messages': messages,
            "model": "modelo_vacio"
        }
        
        response = requests.post(f'{self.base_url}/v1/chat/completions', json=payload, headers=headers)

        if response.status_code == 200:
            finish_reason = response.json()["choices"][0].get("finish_reason")
            #print(f"La generación se detuvo por el siguiente motivo: {finish_reason}")
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f'Error en la solicitud: {response.status_code} - {response.text}')



    def generate_response(self, context: dict, question: str, metadata: dict = None, temperature=0, max_tokens=600) -> str:
        try:
            # Recuperar contexto relevante del documento
            retrieved_context, retrieved_docs = get_documents(question, self.retriever)
            
            if not retrieved_context:
                return "No se encontró contenido relevante en el documento para esta consulta."
            
            # Construir el prompt para el modelo
            prompt_user = (
                f"Contexto del documento:\n{retrieved_context}\n\n"
                f"Datos del paciente: {context.get('data', '')}\n\n"
                f"Pregunta: {question}\n\n"
                "Instrucciones de Formato:\n"
                "Por favor, responde en párrafos separados para mayor legibilidad. "
                "Coloca las referencias en un párrafo separado al final bajo el título 'Referencias'. "
                "Cada referencia debe estar en una nueva línea y numerada.\n\n"
                "Ejemplo de respuesta:\n"
                "Respuesta:\n"
                "Párrafo 1...\n\n"
                "Párrafo 2...\n\n"
                "Referencias:\n"
                "- [1] Documento proporcionado (Página X).\n"
                "- [2] Fuente adicional (Página Y).\n\n"
            )
            
            messages = [
                {"role": "system", "content": self.sys_message},
                {"role": "user", "content": prompt_user}
            ]
            
            # Obtener respuesta generada por el modelo
            generated_text = self.send_chat(messages)
            
            # Procesar y devolver el texto en formato de texto plano
            plain_text_response = self.process_plain_text(generated_text, retrieved_docs)
            return plain_text_response

        except Exception as e:
            return f"Error al generar la respuesta: {str(e)}"
    
    def process_plain_text(self, generated_text, retrieved_docs):
            # Dividir el texto en párrafos utilizando saltos de línea dobles
            paragraphs = generated_text.split('\n\n')

            # Formatear cada párrafo con un salto de línea doble para mayor legibilidad
            formatted_text = "\n\n".join(paragraphs)

            # Procesar las referencias y formatearlas en texto plano
            references = []
            for doc in retrieved_docs:
                page = doc.get("page", "Desconocida")
                filename = doc.get("filename", "").strip()
                if filename and filename != "Documento no identificado":
                    reference = f"{filename} - Página {page}"
                    if reference not in references:
                        references.append(reference)

            # Agregar la bibliografía si existen referencias válidas
            if references:
                bibliography = "\n\nReferencias:\n" + "\n".join([f"- {ref}" for ref in references])
            else:
                bibliography = ""

            # Combinar respuesta formateada y referencias en texto plano
            return formatted_text + bibliography


model1 = tryload(model1_filename)
assistant = BiomedicalAssistant(base_url="http://172.20.8.120:2345", retriever= faiss_retriever)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler

# Rutas de Flask
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Obteniendo datos del formulario...")

        # Obtener datos del formulario
        patient_data = {
            'Age': request.form.get('age'),
            'Sex': request.form.get('sex'),
            'ChestPainType': request.form.get('cp'),
            'RestingBP': request.form.get('resting_bp'),
            'Cholesterol': request.form.get('cholesterol'),
            'FastingBS': request.form.get('fasting_bs'),
            'RestingECG': request.form.get('resting_ecg'),
            'MaxHR': request.form.get('max_hr'),
            'ExerciseAngina': request.form.get('exercise_angina'),
            'Oldpeak': request.form.get('oldpeak'),
            'ST_Slope': request.form.get('st_slope')
        }
        print(f"Datos del formulario recibidos: {patient_data}")

        # Convertir a DataFrame y asegurar que todos los valores son numéricos
        patient_df = pd.DataFrame([patient_data])
        patient_df = patient_df.apply(pd.to_numeric, errors='coerce')

        # Cargar el scaler en esta ruta específica
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Confirmar el tipo de scaler
        print(f"Tipo de scaler después de cargar: {type(scaler)}")
        if not isinstance(scaler, StandardScaler):
            raise ValueError("El objeto cargado no es una instancia de StandardScaler")

        # Transformar los datos del paciente con el scaler
        print("Transformando los datos con el scaler cargado en tiempo real...")
        patient_scaled = scaler.transform(patient_df)
        print("Datos escalados:", patient_scaled)

        # Realizar la predicción
        prediction = model1.predict(patient_scaled)[0]
        prediction_text = 'Alto riesgo' if prediction == 1 else 'Bajo riesgo'

        return jsonify({'prediction': prediction_text})

    except Exception as e:
        print("Error en la predicción:")
        traceback.print_exc()  # Imprime la traza completa del error
        return jsonify({'error': 'Ocurrió un error al predecir el riesgo.'}), 500
from flask import json

@app.route('/chat', methods=['POST'])
def chat():
    # Forzar la carga de datos en bruto y luego decodificar manualmente
    raw_data = request.get_data(as_text=True)
    print("Raw data received:", raw_data)

    try:
        # Decodificar manualmente el JSON
        data = json.loads(raw_data)
        print("Parsed JSON data:", data)

        # Validación para asegurar que 'message' esté en los datos
        if 'message' not in data:
            return jsonify({'error': 'No se proporcionó una consulta.'}), 400

        # Obtener el mensaje y procesarlo
        message = data['message']
        print("Mensaje recibido:", message)

        # Generar respuesta con el asistente utilizando `generate_response`
        context = {}  # Agrega contexto relevante si es necesario (puedes ajustar aquí)
        response_text = assistant.generate_response(context=context, question=message)  # Genera la respuesta

        # Responder con el texto generado por el asistente
        return jsonify({'response': response_text})

    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e}")
        return jsonify({'error': 'La solicitud no contiene un JSON válido.'}), 400

if __name__ == '__main__':
    app.run(debug=False)
