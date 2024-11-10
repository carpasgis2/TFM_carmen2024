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
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer, util
from metapub import PubMedFetcher
import os
import json
from json import JSONDecodeError

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

from metapub import PubMedFetcher


os.environ["NCBI_API_KEY"] = "386c272f5229817d334a9ffc3919db050408"


# Cargar el modelo de embeddings para similitud semántica
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo ligero de similitud semántica

# Configuración de la API de PubMed
os.environ["NCBI_API_KEY"] = "386c272f5229817d334a9ffc3919db050408"
fetch = PubMedFetcher()

def extract_most_similar_phrase(query, num_articles=5):
    """
    Extrae la frase en la consulta que es más similar a un título de PubMed.

    Args:
        query (str): La consulta del usuario.
        num_articles (int): Número de artículos a recuperar de PubMed.

    Returns:
        str: La frase de la consulta más similar al título en PubMed.
    """
    # Dividir la consulta en frases
    query_phrases = query.split('. ')
    
    # Realizar búsqueda en PubMed
    keyword = query  # O extraer una palabra clave específica
    pmids = fetch.pmids_for_query(keyword, retmax=num_articles)
    if not pmids:
        print(f"No se encontraron artículos para '{keyword}' en PubMed.")
        return None

    # Extraer títulos de los artículos recuperados
    pubmed_titles = []
    for pmid in pmids:
        article = fetch.article_by_pmid(pmid)
        if article and article.title:
            pubmed_titles.append(article.title)

    # Calcular la similitud entre cada frase de la consulta y cada título de PubMed
    max_similarity = 0
    most_similar_phrase = None

    for phrase in query_phrases:
        phrase_embedding = semantic_model.encode(phrase, convert_to_tensor=True)
        
        for title in pubmed_titles:
            title_embedding = semantic_model.encode(title, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(phrase_embedding, title_embedding).item()
            
            if similarity_score > max_similarity:
                max_similarity = similarity_score
                most_similar_phrase = phrase

    print(f"Frase más similar: '{most_similar_phrase}' (Similitud: {max_similarity:.2f})")
    return most_similar_phrase


#model = SentenceTransformer('all-MiniLM-L6-v2')

def generar_referencias_pubmed(articles_data):
    """
    Genera una lista de referencias bibliográficas a partir de los datos de artículos obtenidos de PubMed.

    Args:
        articles_data (list): Lista de artículos con datos obtenidos de PubMed.

    Returns:
        list: Lista de referencias formateadas.
    """
    referencias = []
    for article in articles_data:
        if isinstance(article, dict) and article.get("Title"):
            autores = ', '.join(article.get("Author", ["Autores no especificados"]))
            year = article.get("Year", "Año no especificado")
            title = article.get("Title", "Título no disponible")
            journal = article.get("Journal", "Revista no especificada")
            volume = article.get("Volume", "Volumen no especificado")
            issue = article.get("Issue", "Número no especificado")
            link = article.get("Link", "#")

            referencia = (
                f'- {autores} ({year}). "{title}" en {journal} '
                f'(Vol. {volume}, N.º {issue}). '
                f'Disponible en: [PubMed]({link}).'
            )
            referencias.append(referencia)
        else:
            referencias.append("- Información del artículo no disponible en PubMed")
    
    return referencias




import time

def find_pubmed(keywords, num_of_articles=5, retries=3):
    """
    Realiza una búsqueda en PubMed utilizando palabras clave y recupera los artículos más relevantes.
    
    Args:
        keywords (list): Lista de palabras clave para la búsqueda en PubMed.
        num_of_articles (int): Número de artículos a recuperar por palabra clave.
        retries (int): Número de reintentos en caso de error.

    Returns:
        list: Lista de artículos con datos bibliográficos obtenidos de PubMed.
    """
    api_key = os.getenv("NCBI_API_KEY")
    if not api_key:
        print("⚠️ La API key no está configurada en el entorno.")
        return []

    articles_data = []
    for keyword in keywords:
        attempts = 0
        while attempts < retries:
            try:
                pmids = fetch.pmids_for_query(keyword, retmax=num_of_articles)
                if not pmids:
                    print(f"❌ No se encontraron artículos para '{keyword}' en PubMed.")
                    break

                for pmid in pmids:
                    article = fetch.article_by_pmid(pmid)
                    if article:
                        articles_data.append({
                            "pmid": pmid,
                            "Title": article.title,
                            "Abstract": article.abstract,
                            "Author": article.authors,
                            "Year": article.year,
                            "Volume": article.volume,
                            "Issue": article.issue,
                            "Journal": article.journal,
                            "Citation": article.citation,
                            "Link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        })
                break  # Exit retry loop if successful
            except Exception as e:
                print(f"⚠️ Error al recuperar artículos para '{keyword}': {e}. Reintentando ({attempts + 1}/{retries})...")
                time.sleep(2)  # Espera 2 segundos antes de intentar nuevamente
                attempts += 1

    return articles_data


def generar_referencias_completas(referencias_generadas_por_modelo, referencias_pubmed):
    """
    Combina las referencias generadas por el modelo y las referencias obtenidas de PubMed en un solo texto.

    Args:
        referencias_generadas_por_modelo (list): Lista de referencias generadas originalmente.
        referencias_pubmed (list): Lista de referencias relacionadas obtenidas de PubMed.

    Returns:
        str: Texto con ambas secciones de referencias combinadas.
    """
    referencias_texto = "Referencias:\n"
    referencias_texto += "\n".join(referencias_generadas_por_modelo)
    
    referencias_texto += "\n\nBibliografía recomendada por PubMed:\n"
    referencias_texto += "\n".join(referencias_pubmed)
    
    return referencias_texto


def consulta_pubmed(question, num_of_articles=5):
    # Extraer palabras clave de la pregunta utilizando el modelo
    keywords = extract_keywords_with_model(question)
    
    # Buscar artículos en PubMed usando las palabras clave
    articles_data = find_pubmed(keywords, num_of_articles=num_of_articles)
    
    # Generar referencias bibliográficas
    referencias_pubmed = generar_referencias_pubmed(articles_data)

    if referencias_pubmed:
        print("Referencias generadas de PubMed:")
        print(referencias_pubmed)
    else:
        print("No se encontraron referencias para la búsqueda.")
    
    return referencias_pubmed









def extract_keywords_with_model(question):
    # Definir el prompt para pedirle al modelo que extraiga palabras clave
    prompt = f"Por favor, extrae las palabras clave médicas más relevantes de la siguiente pregunta. Lista las palabras clave separadas por comas.\n\nPregunta: {question}\n\nPalabras clave:"
    
    # Preparar los mensajes para el asistente
    messages = [
        {"role": "system", "content": "Eres un asistente que extrae palabras clave médicas de las preguntas de los usuarios."},
        {"role": "user", "content": prompt}
    ]
    
    # Utilizar la función send_chat del asistente para obtener las palabras clave
    keywords_text = assistant.send_chat(messages)
    
    # Procesar la respuesta para obtener una lista de palabras clave
    keywords = [word.strip() for word in keywords_text.split(',') if word.strip()]
    
    return keywords







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

import os
os.environ['FAISS_NO_AVX2'] = '1'

import faiss
print(faiss.__version__)


# Cargar o crear el índice FAISS
# faiss_index_filename = 'faiss_indexML2.pkl'
# if os.path.exists(faiss_index_filename):
#     with open(faiss_index_filename, 'rb') as f:
#         faiss_index = pickle.load(f)
# else:
#     pdf_path = "Manual_AMIR_12da_ed_Cardiologia_y_Cirugi.pdf"
#     document = load_document_pypdf2(pdf_path)
#     chunks = split_documents(document)
#     faiss_index = FAISS.from_documents(chunks, LangChain_Embeddings(model_embeddings))
#     save(faiss_index_filename, faiss_index)

import faiss
from langchain.vectorstores import FAISS


faiss_index = faiss.read_index('faiss_indexML2.idx')
model_name = "bert-base-uncased"  # Cambia esto si necesitas otro modelo
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model_embeddings = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Función para cargar y procesar el archivo PDF
pdf_path = "Manual_AMIR_12da_ed_Cardiologia_y_Cirugi.pdf"
documents = load_document_pypdf2(pdf_path)
chunks = split_documents(documents)

import numpy as np 
# Clase de wrapper para el modelo de embeddings
class BertModelWrapper(Embeddings):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed_documents(self, texts):
        # Generar embeddings para una lista de textos
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Tomar el promedio del último estado oculto
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
        return np.array(embeddings)

    def embed_query(self, text):
        # Generar embeddings para una consulta individual
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Promedio del último estado oculto
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return np.array(embedding).reshape(1, -1)

# Crear una instancia de BertModelWrapper con el modelo cargado
embeddings_model = BertModelWrapper(model_embeddings, tokenizer)

# Crear el vector store FAISS con los fragmentos y el modelo de embeddings
faiss_store = FAISS.from_documents(chunks, embeddings_model)

# Configurar el "retriever" para buscar documentos
faiss_retriever = faiss_store.as_retriever(search_kwargs={'k': 10})




# Configurar el buscador con FAISS
faiss_retriever = faiss_store.as_retriever(search_kwargs={'k': 10})

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
Referencias: Al final de la respuesta, incluye una sección titulada "Referencias:" en un párrafo separado. Enumera las referencias en este formato: "- [1] Documento proporcionado (Página X)." Solo usa referencias directas del documento y, si no hay, utiliza el nombre de la fuente científica, indicando el artículo o fuente específica EN FORMATO APA Y EN INGLÉS.

Respuestas Específicas:
Si la consulta es "¿qué enfermedad tiene el paciente?", utiliza exclusivamente los datos proporcionados en "Datos del paciente".
Si la pregunta es "¿cuáles son los últimos avances...?", busca únicamente en artículos recientes de PubMed.
"""

    def send_request(self, prompt):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        payload = {'prompt': prompt, 'system_message': self.sys_message}
        response = requests.post(f'{self.base_url}/chat', json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f'Error en la solicitud: {response.status_code} - {response.text}')
            
    def send_chat(self, messages):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        payload = {'messages': messages, "model": "modelo_vacio"}
        response = requests.post(f'{self.base_url}/v1/chat/completions', json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f'Error en la solicitud: {response.status_code} - {response.text}')
    
    def generate_references(self, user_query):
        # Esta función debe ser definida para generar referencias desde el contenido del documento proporcionado.
        return []  # Placeholder hasta implementar la generación de referencias.
        
    def extract_keywords_with_model(self, user_query):
        # Prompt ajustado para pedir únicamente palabras clave
        prompt = (
            f"Extrae solo las palabras clave médicas más relevantes de la siguiente pregunta. "
            f"Lista las palabras clave en formato simple, separadas por comas, y no incluyas detalles adicionales:\n\n"
            f"Pregunta: {user_query}\n\n"
            f"Palabras clave:"
        )
        
        # Utilizar el modelo para obtener la lista de palabras clave
        keywords_text = self.send_chat([{"role": "user", "content": prompt}])
        
        # Procesar la respuesta para extraer las palabras clave en una lista
        keywords = [word.strip() for word in keywords_text.split(',') if word.strip()]
        return keywords


    def find_pubmed(self, keywords, num_of_articles=5):
        # Buscar en PubMed con las palabras clave
        articles_data = []
        for keyword in keywords:
            try:
                pmids = fetch.pmids_for_query(keyword, retmax=num_of_articles)
                for pmid in pmids:
                    article = fetch.article_by_pmid(pmid)
                    if article:
                        articles_data.append({
                            "pmid": pmid,
                            "Title": article.title,
                            "Abstract": article.abstract,
                            "Author": article.authors,
                            "Year": article.year,
                            "Volume": article.volume,
                            "Issue": article.issue,
                            "Journal": article.journal,
                            "Citation": article.citation,
                            "Link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        })
            except Exception as e:
                print(f"Error al recuperar artículos para '{keyword}': {e}")
        return articles_data

    def generar_referencias_pubmed(self, articles_data):
        referencias = []
        for article in articles_data:
            autores = ', '.join(article.get("Author", ["Autores no especificados"]))
            year = article.get("Year", "Año no especificado")
            title = article.get("Title", "Título no disponible")
            journal = article.get("Journal", "Revista no especificada")
            link = article.get("Link", "#")
            referencia = f'- {autores} ({year}). "{title}" en {journal}. Disponible en: [PubMed]({link}).'
            referencias.append(referencia)
        return referencias
    
    def generar_referencias_completas(self, referencias_documento, referencias_pubmed):
        texto_referencias = "Referencias:\n" + "\n".join(referencias_documento)
        texto_referencias += "\n\nBibliografía recomendada por PubMed:\n" + "\n".join(referencias_pubmed)
        return texto_referencias

    def generate_response(self, context: dict, user_query: str) -> str:
        try:
            retrieved_context, retrieved_docs = get_documents(user_query, self.retriever)
            if not retrieved_context:
                return "No se encontró contenido relevante en el documento para esta consulta."

            prompt_user = (
                f"Contexto del documento:\n{retrieved_context}\n\n"
                f"Datos del paciente: {context.get('data', '')}\n\n"
                f"Pregunta: {user_query}\n\n"
                "Instrucciones de Formato:\n"
                "Responde en párrafos separados. Al final, coloca 'Referencias' con las referencias extraídas, "
                "y luego 'Bibliografía recomendada por PubMed' con los artículos relacionados.\n\n"
            )
            messages = [{"role": "system", "content": self.sys_message}, {"role": "user", "content": prompt_user}]
            generated_text = self.send_chat(messages)
            return self.process_plain_text(generated_text, retrieved_docs)
        except Exception as e:
            return f"Error al generar la respuesta: {str(e)}"
    
    def process_plain_text(self, generated_text, retrieved_docs):
        return generated_text.strip()


# Ejecución de la lógica con la consulta recibida
def chat():
    raw_data = request.get_data(as_text=True)
    
    try:
        data = json.loads(raw_data)
        if 'message' not in data:
            return jsonify({'error': 'No se proporcionó una consulta.'}), 400

        user_query = data['message']
        print("Mensaje recibido:", user_query)

        # Generar referencias del contenido relevante
        referencias_documento = assistant.generate_references(user_query)

        # Extraer palabras clave
        keywords = assistant.extract_keywords_with_model(user_query)
        print(f"🔍 Palabras clave extraídas para PubMed: {keywords}")

        # Buscar artículos en PubMed
        articles_data = assistant.find_pubmed(keywords, num_of_articles=5)

        # Generar la bibliografía recomendada por PubMed
        referencias_pubmed = assistant.generar_referencias_pubmed(articles_data)

        # Combinar las referencias
        referencias_completas = assistant.generar_referencias_completas(referencias_documento, referencias_pubmed)

        # Crear contexto
        context = {"data": user_query, "referencias": referencias_completas}

        # Generar la respuesta final
        response_text = assistant.generate_response(context=context, user_query=user_query)

        return jsonify({'response': response_text})

    except json.JSONDecodeError:
        return jsonify({'error': 'La solicitud no contiene un JSON válido.'}), 400



model1 = tryload(model1_filename)
assistant = BiomedicalAssistant(base_url="http://172.20.8.120:2345", retriever= faiss_retriever)



scaler = StandardScaler
import numpy as np 
def predict_heart_disease(model1, patient_data):
    try:
        # Realizar la predicción
        prediction = model1.predict(patient_data)
        print("Resultado de la predicción (crudo):", prediction)  # Imprime la salida original

        # Verifica si es un array y si tiene el valor esperado (0 o 1)
        if isinstance(prediction, (list, np.ndarray)):
            prediction_value = prediction[0]
            print("Valor de predicción:", prediction_value)  # Imprime el valor de predicción específico
            return prediction_value
        else:
            print("Formato de predicción inesperado:", prediction)
            return None  # Devuelve None si el formato es inesperado
    except Exception as e:
        print("Error en la predicción del modelo:", str(e))
        return None



# Rutas de Flask
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/buscar_pubmed', methods=['GET'])
def buscar_pubmed_route():
    keyword = request.args.get('keyword', 'cardiovascular disease')
    num_of_articles = int(request.args.get('num_of_articles', 10))
    print(f"🔍 Realizando búsqueda en PubMed con '{keyword}' y {num_of_articles} artículos")  # Debug
    articles = find_pubmed(keyword, num_of_articles)
    
    # Comprobar si se han encontrado artículos
    if articles:
        print(f"✅ {len(articles)} artículos encontrados")  # Debug
    else:
        print("⚠️ No se encontraron artículos o la API falló")  # Debug
    
    return jsonify(articles)

@app.route('/consulta_pubmed', methods=['GET'])
def consulta_pubmed():
    keyword = request.args.get('keyword', 'cardiovascular disease')
    num_of_articles = int(request.args.get('num_of_articles', 10))

    # Realizar la búsqueda en PubMed
    articles_data = find_pubmed(keyword, num_of_articles)

    # Generar referencias con datos reales
    referencias = generar_referencias_pubmed(articles_data)
    return jsonify({
        "referencias": referencias
    })

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
        patient_df = patient_df.astype('float64')
        

        # Cargar el scaler en esta ruta específica
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        


        patient_scaled = scaler.transform(patient_df)

        # Realizar la predicción
        prediccion = predict_heart_disease(model1, patient_scaled)
        if prediccion == 1:
            prediction_text = 'Alto riesgo de enfermedad cardíaca'
        elif prediccion == 0:
            prediction_text = 'Bajo riesgo de enfermedad cardíaca'
        else:
            prediction_text = 'Error: resultado inesperado de la predicción'


        return jsonify({'prediction': prediction_text})

    except Exception as e:
        print("Error en la predicción:")
        traceback.print_exc()  # Imprime la traza completa del error
        return jsonify({'error': 'Ocurrió un error al predecir el riesgo.'}), 500
from flask import json
import json 

@app.route('/chat', methods=['POST'])
def chat():
    raw_data = request.get_data(as_text=True)
    
    try:
        data = json.loads(raw_data)
        if 'message' not in data:
            return jsonify({'error': 'No se proporcionó una consulta.'}), 400

        user_query = data['message']
        print("Mensaje recibido:", user_query)

        # Generar referencias del contenido relevante
        referencias_documento = assistant.generate_references(user_query)

        # Extraer palabras clave
        keywords = assistant.extract_keywords_with_model(user_query)
        print(f"🔍 Palabras clave extraídas para PubMed: {keywords}")

        # Buscar artículos en PubMed
        articles_data = assistant.find_pubmed(keywords, num_of_articles=5)

        # Generar la bibliografía recomendada por PubMed
        referencias_pubmed = assistant.generar_referencias_pubmed(articles_data)

        # Combinar las referencias
        referencias_completas = assistant.generar_referencias_completas(referencias_documento, referencias_pubmed)

        # Crear contexto
        context = {"data": user_query, "referencias": referencias_completas}

        # Generar la respuesta final
        response_text = assistant.generate_response(context=context, user_query=user_query)

        return jsonify({'response': response_text})

    except json.JSONDecodeError:
        return jsonify({'error': 'La solicitud no contiene un JSON válido.'}), 400
    
if __name__ == '__main__':
    app.run(debug=False)
