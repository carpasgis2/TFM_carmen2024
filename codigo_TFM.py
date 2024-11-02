import pandas as pd

def read_csv(csv_file, new_csv_file=None):

    df = pd.read_csv(csv_file)
    
    if new_csv_file:
        df.to_csv(new_csv_file, index=False)
        print(f"Archivo CSV guardado como: {new_csv_file}")
    else:
        print(df)

# Especifica el archivo de entrada y salida
csv_file = "dataset_TFM.xls"
new_csv_file = 'dataset_TFM.csv'  

read_csv(csv_file, new_csv_file)

### CARGA DE LIBRERIAS Y FUNCIONES ACCESORIAS##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Configuración de estilo de colores para las gráficas
sns.set_theme(style="whitegrid")
sns.set_palette(['black', 'gray', 'red'])  # Paleta de colores rojo, blanco, negro y gris

# Función para graficar la importancia de las características
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    # Graficar la importancia de las características
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette=['gray', 'black', 'red'])
    plt.title('Importancia de las características para predecir la enfermedad cardíaca')
    plt.tight_layout()
    plt.show()

model1_filename = 'modelo_entrenado.pkl'

if not os.path.exists(model1_filename):
    df = pd.read_csv("dataset_TFM.csv")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Mapeo de variables categóricas a valores numéricos
    mapping_dict = {
        'Sex': {'M': 1, 'F': 0},
        'ChestPainType': {'ATA': 1, 'NAP': 2, 'ASY': 3, 'TA': 4},
        'RestingECG': {'Normal': 1, 'ST': 2, 'LVH': 3},
        'ExerciseAngina': {'Y': 1, 'N': 0},
        'ST_Slope': {'Up': 1, 'Flat': 2, 'Down': 3}
    }
    df.replace(mapping_dict, inplace=True)

    # Imprimir estadísticas del dataset
    print(df.describe())
    print(f"Valores duplicados: {df.duplicated().sum()}")
    print(f"Valores faltantes:\n{df.isna().sum()}")

    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Gráfico de calor con la descripción del dataset
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.describe(), annot=True, cmap='Greys')
    plt.title('Descripción del conjunto de datos de ataques cardíacos')
    plt.show()

    # Histograma de todas las variables
    df.hist(bins=20, figsize=(15, 10), color='gray', edgecolor='black')
    plt.show()

    # Gráficos de barras para las principales variables categóricas
    columns = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    target_column = 'HeartDisease'
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(columns, 1):
        plt.subplot(3, 3, i)
        sns.countplot(data=df, x=column, hue=target_column, palette=['gray', 'red'])
        plt.title(f'Relación entre {column} y ataques cardíacos ({target_column})')
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.legend(title="HeartDisease", loc='upper right')
    plt.tight_layout()
    plt.show()

    # Histograma para variables numéricas seleccionadas con ajustes para legibilidad
    num_columns = ['Cholesterol', 'MaxHR']
    plt.figure(figsize=(12, 5))
    for i, column in enumerate(num_columns, 1):
        plt.subplot(1, 2, i)
        sns.histplot(data=df, x=column, hue=target_column, element="step", palette=['gray', 'red'], bins=20, kde=True)
        plt.title(f'Relación entre {column} y ataques cardíacos ({target_column})')
        plt.xlabel(column)
        plt.ylabel("Frecuencia")
        plt.legend(title="HeartDisease", loc='upper right')
    plt.tight_layout()
    plt.show()

    # Separar datos y objetivo
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar el modelo
    model1 = RandomForestClassifier()
    model1.fit(X_train_scaled, y_train)

    # Validación cruzada (5-fold)
    scores = cross_val_score(model1, X_train_scaled, y_train, cv=5)
    print(f"Cross-Validation Accuracy: {scores.mean():.4f}")

    # Evaluar el modelo
    y_pred = model1.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy en el conjunto de prueba: {accuracy:.4f}")

    # Guardar el modelo entrenado
    with open(model1_filename, 'wb') as f:
        pickle.dump(model1, f)

    print(f"Modelo entrenado y guardado en {model1_filename}")

    # Graficar la importancia de las características
    plot_feature_importance(model1, X.columns)
    
else:
    # Cargar el modelo entrenado si ya existe
    print(f"El archivo {model1_filename} ya existe. No es necesario volver a entrenar el modelo.")
    with open(model1_filename, 'rb') as f:
        model1 = pickle.load(f)

    # Asumiendo que el CSV ya ha sido cargado y preparado previamente
    df = pd.read_csv("dataset_TFM.csv")
    target_column = 'HeartDisease'
    X = df.drop(target_column, axis=1)

    # Graficar la importancia de las características
    plot_feature_importance(model1, X.columns)


##INTRODUCIR EL IMPUT PARA LA PREDICCIÓN DEL MODELO DE MACHINE LEARNING##

def get_patient_data():
    try:
        age = input("Edad (20-100): ") or None
        sex = input("Sexo (0=Femenino, 1=Masculino): ") or None
        cp = input("Tipo de dolor en el pecho (0=Asintomático, 1=Angina atípica, 2=Angina típica, 3=No anginoso): ") or None
        resting_bp = input("Presión arterial en reposo (90-200 mmHg): ") or None
        cholesterol = input("Colesterol sérico (100-500 mg/dl): ") or None
        fasting_bs = input("Azúcar en sangre en ayunas (0=No, 1=Sí): ") or None
        resting_ecg = input("Resultados electrocardiográficos en reposo (0=Normal, 1=Anormalidad en ST-T, 2=Hipertrofia ventricular izquierda): ") or None
        max_hr = input("Frecuencia cardíaca máxima alcanzada (60-200 ppm): ") or None
        exercise_angina = input("Angina inducida por ejercicio (0=No, 1=Sí): ") or None
        oldpeak = input("Depresión ST (0-6): ") or None
        st_slope = input("Pendiente del segmento ST (0=Ascendente, 1=Plana, 2=Descendente): ") or None

        patient_data = {
            'Age': age,
            'Sex': sex,
            'ChestPainType': cp,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'RestingECG': resting_ecg,
            'MaxHR': max_hr,
            'ExerciseAngina': exercise_angina,
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope
        }

        patient_df = pd.DataFrame([patient_data])
        patient_df = patient_df.apply(pd.to_numeric, errors='coerce')
        return patient_df, f"Datos del paciente: {patient_data}"

    except Exception as e:
        print(f"Error al ingresar datos: {e}")
        return None, None



def predict_heart_disease(model1, patient_data):
    try:
        prediction = model1.predict(patient_data)[0]
        return 'Alto riesgo' if prediction == 1 else 'Bajo riesgo'
    except Exception as e:
        return f"No se pudo realizar la predicción debido a un error: {e}"



from langchain.document_loaders import PyPDFLoader 

def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    print(f"Archivo guardado correctamente en {file_name}")

def tryload(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, AttributeError) as e:
        print(f"Error al cargar {file_name}: {e}")
        return None

##CARGA DE DOCUMENTOS Y FRAGMENTACIÓN##

import PyPDF2

def load_document_pypdf2(pdf_path):
    documents = []

    try:
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)

            total_pages = len(reader.pages)
            print(f"Total de páginas detectadas: {total_pages}")

            prelim_pages = 0

            for page_number in range(total_pages):
                page = reader.pages[page_number]
                page_content = page.extract_text()

                if page_content:

                    visible_page_number = page_number + 1  
                    adjusted_page_number = visible_page_number - prelim_pages  


                    if adjusted_page_number > 0:
                        doc = {
                            "page_content": page_content.strip(),
                            "metadata": {
                                "filename": pdf_path.split('\\')[-1],
                                "page": adjusted_page_number  
                            }
                        }
                        documents.append(doc)

            for doc in documents:
                if doc['metadata']['page'] == 30:
                    print(f"Contenido de la página {doc['metadata']['page']}:")
                    print(doc['page_content'])
                    print("\nMetadatos:")
                    print(doc['metadata'])
                    break
            else:
                print("No se encontró la página 30.")

    except Exception as e:
        print(f"No se pudo cargar el archivo PDF: {e}")
    
    return documents

# Llamar a la función con el archivo PDF
pdf_path = "Manual_AMIR_12da_ed_Cardiologia_y_Cirugi.pdf"
load_document_pypdf2(pdf_path)




DATA_PATH = pdf_path 

document = tryload("documentML2.pkl")
if not document:
    document = load_document_pypdf2(DATA_PATH)
    save("documentML2.pkl", document)



def split_documents(document: list):
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,   
        chunk_overlap=120,  
        length_function=len,
        is_separator_regex=False,
    )
    for doc in document:
        if isinstance(doc, dict):
            doc = Document(
                page_content=doc.get('page_content', ''),
                metadata=doc.get('metadata', {})
            )
        split_texts = text_splitter.split_text(doc.page_content)
        for chunk in split_texts:
                new_doc = Document(
                    page_content=chunk,
                    metadata=doc.metadata  
                )
                chunks.append(new_doc)
    return chunks




chunks = tryload("chunksML2.pkl")
if not chunks:
    chunks = split_documents(document)
    save("chunksML2.pkl", chunks)

# Importaciones de PyTorch y Transformers
import torch
from transformers import AutoModel, AutoTokenizer

# Importaciones de FAISS y LangChain para el índice de búsqueda semántica
import faiss
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cargar y guardar archivos (asegúrate de definir tryload y save)
import pickle
import os

device = torch.device("cpu")

##CREAR REPRESENTACIONES VECTORIALES DEL TEXTO Y ALMACENARLO##

class LangChain_Embeddings(Embeddings):
    def __init__(self, embedder: AutoModel):
        self.embedder = embedder
        super().__init__()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embeddings = self.embedder(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> list[float]:
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=200)
        with torch.no_grad():
            embedding = self.embedder(**inputs).last_hidden_state.mean(dim=1)
        return embedding.cpu().numpy().tolist()[0]
    




model_name = "jinaai/jina-embeddings-v2-base-es"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.to(device)



faiss_index = tryload("faiss_indexML2.pkl")
if not faiss_index:
    faiss_index = FAISS.from_documents(chunks, LangChain_Embeddings(model))
    save("faiss_indexML2.pkl", faiss_index)



##RAG##
faiss_retriever = faiss_index.as_retriever(search_kwargs={'k': 10})

def get_documents(query, retriever):
    unique_docs = retriever.invoke(query)
    documents_with_pages = []
    context = ""
    for doc in unique_docs:
        page_info = doc.metadata.get('page', 'Página desconocida')
        document_filename = doc.metadata.get('filename', 'Documento no identificado')
        # Incluir el número de página en el contexto
        content = f"[Página {page_info}]\n{doc.page_content}\n\n"
       
        context += content
        documents_with_pages.append({
            "page": page_info,
            "content": doc.page_content,
            "filename": document_filename
        })
    if context:
        print("Contenido recuperado del documento:", context)
    else:
        print("No se encontró contenido relacionado con la pregunta en el documento.")
    return context, documents_with_pages


##LLAMAR AL LARGE LENGUAJE MODEL##

import requests

class BiomedicalAssistant:
    def __init__(self, base_url, retriever, api_key=None):
        self.base_url = base_url
        self.retriever = retriever
        self.api_key = api_key
        self.sys_message = '''
Eres un asistente especializado en enfermedades cardíacas. Tu tarea es proporcionar respuestas claras, objetivas y fundamentadas en los resultados médicos y el contenido del documento proporcionado.

Debes ofrecer interpretaciones precisas de datos clínicos y diagnósticos en condiciones como infarto de miocardio, hipertensión, arritmias y otros problemas cardiovasculares.

Las respuestas deben estar basadas únicamente en el documento indicado, usando referencias entre corchetes numerados (por ejemplo, [1], [2]) y proporcionando una lista de referencias al final de cada respuesta. Las referencias deben contener la página y número de párrafo de donde se ha extraído la respuesta. 
Usa los números de página tal como aparecen en el contexto (por ejemplo, "[Página 30]") y no inventes números de página.

Evita incluir información redundante o fragmentos innecesarios, a menos que estén directamente relacionados con la pregunta.

En el caso de que la respuesta esté en varias páginas, pon todas las páginas usadas. NO INCLUYAS LA REFERENCIA MIR, sino la página.

Ante la pregunta de: qué enfermedad tiene el paciente, básate únicamente en los datos proporcionados en: "Datos del paciente".

Si hago una pregunta de algo que no está en el documento, contesta igualmente con la información disponible de documentos científicos y pon las referencias como: nombre del artículo.

Cuando se te prefgunte por: "¿cuales son los ultimos avances de...." busca EXCLUSIVAMENTE en artículos de PubMed para dar una repuesta, poriorizando SIEMPRE LOS ARTÍCULOS MÁS RECIENTES

NO TE INVENTES BIUBLIOGRTAFÍA
'''

    
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
            retrieved_context, _ = get_documents(question, self.retriever)
            if not retrieved_context:
                return "No se encontró contenido relevante en el documento para esta consulta."
            
            prompt_user = f"Contexto del documento:\n{retrieved_context}\n\nDatos del paciente: {context.get('data', '')}\n\nPregunta: {question}"
            print(f"Prompt para el modelo:\n{prompt_user}")
            
            messages = [
                {"role": "system", "content": self.sys_message},
                {"role": "user", "content": prompt_user}
            ]
            
            generated_text = self.send_chat(messages)
            return generated_text

        except Exception as e:
            return f"Error al generar la respuesta: {str(e)}"




model1 = tryload(model1_filename)
assistant = BiomedicalAssistant(base_url="http://172.20.8.120:2345", retriever= faiss_retriever)


def main():
    print("--------------------------------------------------------------------------")
    print("Bienvenido al Asistente Biomédico Virtual. Primero ingresa los datos del paciente para hacer una predicción.\n")
    patient_data, context = get_patient_data()
    if patient_data is not None:
        print("Datos del paciente ingresados correctamente.\n")
        prediccion = predict_heart_disease(model1, patient_data)
        print(f"Predicción: {prediccion} de enfermedad cardíaca.\n")
    else:
        print("No se ingresaron datos del paciente o hubo un error.")
    print("\nPuedes hacerle cualquier pregunta relacionada con la salud cardíaca o escribir 'salir' para finalizar.")
    while True:
        query = input("\n¿Qué deseas saber?:\n")
        
        if query.lower() in ['salir', 'exit', 'quit']:
            print("Gracias por usar el Asistente Biomédico. ¡Cuídate!")
            break
        print("Buscando respuesta...\n")
        
        retrieved_context, retrieved_docs = get_documents(query, faiss_retriever)
        
        if retrieved_docs:
            metadata = {
                "filename": retrieved_docs[0].get('filename', 'Documento no identificado'),
                "page": retrieved_docs[0].get('page', 'Página desconocida')
            }
        else:
            metadata = None
        
        context_dict = {"data": context, "retrieved_context": retrieved_context}
        
        response = assistant.generate_response(context_dict, query, metadata=metadata)
        if response:
            print("Respuesta:\n" + response + "\n")
        else:
            print("No se pudo generar una respuesta válida en este momento.\n")
if __name__ == "__main__":
    main()


##para que se usa el trastuzumab 
## hablame sobre la respuesta de las taquicaridas a las maniobras que deprimen la conducion del nodo AV  #pagina 67
## por que se caractertiza la taquicadia sinusal 
# para que se usa la Propafenona
#. Etiología de la cardiopatía isquémica
#El Síndrome de Brugada es una enfermedad  pag 73



#prerdiccion de efuturos eventos dada una infomracion
#localizacion de informacion tecnica en bbdd muy tecnincas o blibliografia acotada

#dada la informacion apaortada que enfemredad podria desariollar?
#que patologia?
#apollo muy tecnico : que tratamiento me recoemndarias para 



#que es la  amiodarona
#que enfermedad tiene el pacioente
