import pandas as pd
from matplotlib import pyplot as plt
import os
from tabulate import tabulate
from IPython.display import display, Markdown, Latex

import numpy as np
from google import genai
from google.genai import types

from bm25_vectorizer import BM25Vectorizer

from collections.abc import Callable, Sequence
from typing import Any, Literal

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
project_root = next((parent for parent in Path.cwd().resolve().parents
                        if Path(parent, 'notebooks').is_dir()))
sys.path.insert(0, str(project_root))

import re

import chromadb

from dataclasses import dataclass

from tenacity import retry, wait_random_exponential



#### INSTANCIACIÓN CLIENTE GEMINI

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])



#### CARGA DE DATOS

def load_data(file: str, key: set) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(project_root, 'datos', file), sep=',', decimal='.', encoding='utf-8')
    df.set_index(key, inplace=True)
    return df



#### EVALUACIÓN

def recall_at_k(query_id: int, valid_esci_labels: list, ranked_docs: pd.DataFrame, relevances: pd.DataFrame, k:int = 10) -> float:
    """
    calula el recall para una columna de una matriz de ranking (query) dada una lista de etiquetas a considerar y una tabla de relevancia

    Args:
        query_id: identificador de la columna (query)
        valid_esci_labels: lista de los esci labels a considerar; los elementos deben estar entre E, S, C, I
        ranked_docs: la matriz con los rankings (query_id como nombres de columna, product_id como valores ordenados de mayor a menor relevancia)
        relevances: la tabla de relevancias indexada por 'product_id'
        k: evalúa los primeros k regitros de 'ranked_docs'
    """
    pid_list = relevances[(relevances['query_id'] == query_id) & (relevances['esci_label'].isin(valid_esci_labels))].index.to_list()
    x = 0
    retrieved_pid_list = ranked_docs[query_id].to_list()[:k]
    for pid in pid_list:
        if pid in retrieved_pid_list:
            x += 1
    return x/len(pid_list)

def nDCG_at_k(query_id: int, valid_esci_labels: list, ranked_docs: pd.DataFrame, relevances: pd.DataFrame, k:int = 10) -> float:
    """
    calula el nDCG (Discounted Cumulative Gain) para una columna de una matriz de ranking (query) dada una lista de etiquetas a considerar y una tabla de relevancia
    La etiqueta I tiene relevancia 0. Debería plantearse que fuese -1, ya que los resultados relevantes omitidos tienen una relevancia implícita de 0. 
    A nivel de experiéncia de usuario es peor que se omitan resultados a que se devuelvan resultados irrelevantes: un fallo pasa inadvertido, el otro es explícito

    Args:
        query_id: identificador de la columna (query)
        valid_esci_labels: no se usa realmente internamene, pero está por tener uniformidad de parámetros entre todas la métricas, lo que facilita la evaluación conjunta
        ranked_docs: la matriz con los rankings (query_id como nombres de columna, product_id como valores ordenados de mayor a menor relevancia)
        relevances: la tabla de relevancias indexada por 'product_id'
        k: evalúa los primeros k regitros de 'ranked_docs'
    """
    import math
    #relevances_sorted.sort_values(by=['query_id', 'relevance'], ascending=False) # se ordena, de más a menos, para poder calcular después el idcg
    relevances_sorted = relevances[relevances['query_id'] == query_id]['relevance'].sort_values(ascending=False) # se extraen los pid que se corresponden a los valid_esci_labels
    retrieved_pid_list = ranked_docs[query_id].to_list()[:k] # se quitan pid que se corresponden a esci que no están en valid_esci_labels
    dcg = 0
    for i,pid in enumerate(retrieved_pid_list, start=1):
        r = relevances_sorted.loc[pid] if pid in relevances_sorted.index else 0 # si el product_id no está entre las etiquetas de relevances, se considera irrelevante, valor 0
        dcg += r/math.log2(i +1)
    idcg = 0
    for i,relevance in enumerate(relevances_sorted.to_list()[:k], start=1):
        idcg += relevance/math.log2(i +1)
    return dcg/idcg

def MRR_at_k(query_id: int, valid_esci_labels: list, ranked_docs: pd.DataFrame, relevances: pd.DataFrame, k:int = 10) -> float:
    """
    calula el MRR (Mean Reciprocal Rank) para una columna de ranking (query) dada una lista de etiquetas a considerar y una tabla de relevancia

    Args:
        query_id: identificador de la columna (query)
        valid_esci_labels: lista de los esci labels a considerar; los elementos deben estar entre E, S, C, I
        ranked_docs: la matriz con los rankings (query_id como nombres de columna, product_id como valores ordenados de mayor a menor relevancia)
        relevances: la tabla de relevancias indexada por 'product_id'
        k: evalúa los primeros k regitros de 'ranked_docs'
    """
    relevant_pids = relevances[(relevances['query_id']==query_id) & (relevances['esci_label'].isin(valid_esci_labels))].index.to_list()
    retrieved_pids = ranked_docs[query_id].to_list()[:k]
    return next((1/(retrieved_pids.index(pid) +1) for pid in retrieved_pids if pid in relevant_pids), 0) 

def result_analysis(query_id: int, valid_esci_labels: list, ranked_docs: pd.DataFrame, relevances: pd.DataFrame, docs: pd.Series, devq: pd.DataFrame, chop: int = 100):
    """
    para un query_id muestra:
        - el texto de la query
        - el título de las respuestas consideradas válidas según esci_labels que se consideran como respuesta válida, con código producto
        - las los documentos respuesta, con código producto

    Args:
        query_id: identificador de la columna (query)
        valid_esci_labels: lista de los esci labels a considerar; los elementos deben estar entre E, S, C, I
        ranked_docs: la matriz con los rankings
        labeled_query_results: un pd.DataFrame de de resultados etiquetados a las queries de prueba con las que se han obtenido los rangos, con `roduct_id' como índice
        docs: el corpus
        devq: las queries con resultados etiquetados en relevances_sorted
    """
    from html import escape
    from IPython.display import HTML, display

    right_answers = docs.loc[relevances[(relevances['query_id']==query_id) & (relevances['esci_label'].isin(valid_esci_labels))].index]
    retrieved_answers = docs.loc[ranked_docs[query_id]]
    query = devq.loc[query_id, 'query_text']

    display(HTML(f'<b>QUERY:</b><p>{escape(str(query))}</p>'))
    display(HTML('<b>RIGHT ANSWERS (according to selected esci labels):</b>'))
    display(HTML(tabulate(right_answers.str[:chop].to_frame(), headers='keys', tablefmt='html', maxcolwidths=[80])))
    display(HTML('<b>RETRIEVED ANSWERS:</b>'))
    display(HTML(tabulate(retrieved_answers.str[:chop].to_frame(), headers='keys', tablefmt='html', maxcolwidths=[80])))



#### CREACION DE EMBEDDINGS

def text_parser(text):
    """
    parsea un una pd.Series conteniendo los textos para para facilitar la creación de embeddings a partir de ellos. Funciona tanto por columnas como por filas o escalar
        1. se rellenan los campos sin información con cadena vacía
        2. se asegura que todos los campos sean de tipo string
        3. se pasa todo a minúscula
        4. se sustituye cualquier caracter que no sea alfanumérico por un espacio
        5. separa en palabras (el separador es el espacio blanco)
        6. junta de nuevo con un único espacio entre palabaras 
    """
    if isinstance(text, pd.Series):
        return (
            text
            .astype(str) # se asegura que todos los campos sean de tipo string
            .fillna('') # se rellenan los campos sin información con cadena vacía
            .str.lower() # se pasa todo a minúscula
            .str.replace(r'[^a-z0-9À-ÿ\s]', ' ', regex=True) # se sustituye cualquier caracter que no sea alfanumérico por un espacio
            .str.split() # separa en palabras (el separador es el espacio blanco)
            .str.join(' ') # junta de nuevo con un espacio entre palabaras
        )
    elif text:
        text = str(text)
        text = text.lower()
        text = re.sub(r'[^a-z0-9À-ÿ\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    else:
        return ''


def parse_corpus(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    añade aun dataframe que contenga las columnas 'brand', 'title' y 'text' una columna conteniendo un texto optimizado para ser codificado
    
    Args:
        - pandas dataframe con las columnas 'brand', 'title' y 'text' y 'product_id' como índice
    
    Returns:
        - una copia del dataframe de entrada con la columna 'string2embed' añadida
        - 'string2embed' contiene strings para usar para crear los embeddings, f'title: {x.title} | text: {x.text} | marca: {x.brand}'
        Cumplen con el formato esperado por gemini-embedding-2 (f'title: {x.title} | text: {x.text}') para la tarea tipo 'crear embeddings de documemntos'
    """
    #corpus_cp = corpus[corpus['active']==True]
    corpus_cp=corpus.copy()
    if len(corpus_cp.drop_duplicates()) != len(corpus_cp):
        raise ValueError('duplicated active products (duplicates in product_id column after filtering out rows with active==False)')

    #corpus_cp.drop(['color', 'locale', 'catalog_version'], axis=1, inplace=True) # se mantiene 'record_id' para usarlo como ids en la base de datos

    corpus_cp[['brand', 'title', 'text']] = corpus_cp[['brand', 'title', 'text']].astype(str).fillna('')
    brands = corpus_cp['brand'].value_counts().index
    def fill_brand(row):
        return next((brand for brand in brands if brand in row['title'] or brand in row['text']), 'unknown')
    corpus_cp.loc[corpus_cp['brand'].isna(), 'brand'] = corpus_cp[corpus_cp['brand'].isna()].apply(lambda x: fill_brand(x), axis=1)

    corpus_cp[['brand_p', 'title_p', 'text_p']] = corpus_cp[['brand', 'title', 'text']].apply(text_parser, axis=0)  
    corpus_cp['string2embed'] = corpus_cp[['brand_p', 'title_p', 'text_p']].agg(lambda x: f'title: {x.title_p} | text: {x.text_p} | marca: {x.brand_p if x.brand_p!='unknown' else ''}', axis=1) 
    corpus_cp.drop(['brand_p', 'title_p', 'text_p'], axis=1, inplace=True)

    return corpus_cp

@retry(wait=wait_random_exponential(multiplier=1, max=60))
def embed_documents_ge1(
    documents: pd.Series,
    task_type: Literal['RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY'],
    output_dimensionality: Literal[768, 1536, 3072],
    batch_size: int = 100,
) -> np.ndarray:
    """
    genera embeddings con genai-embedding-001

    Args:
        documents: una pd.Series con el listado de textos para crear los embeddings, con product_id como índice
        task_type: tipo de embeddings a generar, si de documento o de query
        batch_size: para no sobrepasar el nivel de concurrencia que espera el modelo. 100 es el valor máximo
        output_dimensionality: posibles dimensiones de recomendadas por google
    
    Returns:
        un np.dnarray con los vectores en filas. El orden sigue el de documents (documento de fila n se correspone a vector de fila n)
    """
    

    if not 1 <= batch_size <= 100:
        raise ValueError('batch_size debe estar entre 1 y 100')

    texts = documents.fillna('').astype(str).tolist() # se asegura sustituir los NA por cadena vacía
    embeddings = []

    for start in range(0, len(texts), batch_size):
        response = client.models.embed_content(
            model='gemini-embedding-001',
            contents=texts[start:start + batch_size],
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality,
            ),
        )
        embeddings.extend(embedding.values for embedding in response.embeddings)

    vectors = np.asarray(embeddings, dtype=np.float32)  

    # si se toman menos dimensiones de las máximas se pierde la normalización 
    if output_dimensionality == 3072:
        return vectors
    else:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, np.finfo(np.float32).eps)

@retry(wait=wait_random_exponential(multiplier=1, max=60))
def embed_documents_ge2(
    documents: pd.Series | str,
    task_type: Literal['RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY'],
    output_dimensionality: Literal[768, 1536, 3072],
    batch_size: int = 100,
) -> np.ndarray:
    """
    genera embeddings con genai-embedding-2

    Args:
        documents: una pd.Series con el listado de textos para crear los embeddings, con product_id como índice
                   si los documentos son para indexar documentos, espera que tengan este formato: title: {title} | text: {content}, if there is no title, use title: none. 
        task_type: tipo de embeddings a generar, si de documento o de query
        batch_size: para no sobrepasar el nivel de concurrencia que espera el modelo. 100 es el valor máximo
        output_dimensionality: posibles dimensiones de recomendadas por google

    Returns:
        un np.dnarray con los vectores en filas. El orden sigue el de documents (documento de fila n se correspone a vector de fila n)
    """
    if not 1 <= batch_size <= 100:
        raise ValueError('batch_size debe estar entre 1 y 100')
    if isinstance(documents, pd.Series): 
        texts = documents.fillna('').astype(str).tolist() # los NA se sustituyen por cadena vacía
    else:
        texts = [str(documents)]
    if task_type == 'RETRIEVAL_QUERY': # si es tipo RETRIEVAL_DOCUMENT ya viene codificado como debe ser (ya se espera), de acuerdo a la tarea
        texts = [f'task: search result | query: {text}' for text in texts]
    if not texts: 
        return np.empty((0, output_dimensionality), dtype=np.float32)

    embeddings = []
    for start in range(0, len(texts), batch_size):
        contents = [
            types.Content(parts=[types.Part.from_text(text=text)]) # es para forzar a generar un embedding por documento
            for text in texts[start:start + batch_size]
        ]
        response = client.models.embed_content(
            model='gemini-embedding-2',
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=output_dimensionality,
            ),
        )
        embeddings.extend(embedding.values for embedding in response.embeddings)
    
    return np.asarray(embeddings, dtype=np.float32) # gemini-embedding-2 normaliza automáticamente, sea cual sea output_dimensionality


#### FUNCIONES PARA COMPARACIÓN EN EL USO DE DIFERENTES TIPOS DE EMBEDDINGS

def run_query_dense(name: str, corpus_embeddings: np.ndarray, query_embeddings: np.ndarray, 
                    corpus_index: list, query_index: list, 
                    k: int = 10) -> pd.DataFrame:
    """
    ejecuta un conjunto de queries sobre un corpus.
    espera vectores normalizados, de forma que la distancia coseno se puede calcular muy eficintentente, dado que es entonces equivalente a un procuto escalar

    Args:
        name: permite identificarla en la tabla de comparación que genera evaluate_compare_queries
        corpus_emdeddings: se esperan vectores normalizados
        query_embeddings: se esperan vectores normalizados y codificados con el mismo algoritmo que el corpus
        corpus_index: se asume que el índice del corpus es la clave y que los vectores están ordenados igual como lo están los docuemntos de corpus, 
            de forma que se puede asignar clave de documento a clave de vector_documento cruzando por posición
        query_index: se asume que el índice de las queries es la clave y que los vectores están ordenados igual como lo están las queries, 
            de forma que se puede asignar clave de query a clave de vector_query cruzando por posición
        rank: se retornan los rank productos más relevantes
    
    Retorna:
        un pd.DataFrame con una columna por query conteniendo los product_id retornados ordenados por relevancia (de mayor a menor score - de menor a mayor distancia coseno -)
        los nombres de coluna son los query_id
    """
    scores = np.dot(corpus_embeddings, query_embeddings.T) # como son vectores normalizados el producto escalar es equivalente a la distancia coseno
    dense_rank = np.argsort(-scores, axis=0)[:k,:] # se se niega scores, de forma que el orden es descendente (el coseno de dos vectores paralelos es 1, ortogonales 0)
    dense_rank_df = pd.DataFrame(dense_rank).apply(lambda x: corpus_index[x])
    dense_rank_df.columns = query_index.to_list()
    dense_rank_df.attrs['name'] = name
    return dense_rank_df

def run_query_sparse(name: str, vectorizer: BM25Vectorizer, queries: pd.Series, corpus_index, k: int = 10) -> pd.DataFrame:
    """
    ejecuta un conjunto de queries sobre un corpus con docificación IDF, calculando el ranking con BM25
    
    Args:
        name: permite identificarla en la tabla de comparación que genera evaluate_compare_queries
        vectorizer: un objeto BM25Vectorizer que contiene la codificación IDF del corpus y los parámetros de BM25
        queries: un pd-Series con query_id como índice. El texto de las queries debe tener el mismo parseo que aplicado al corpus con el que se ha construido el vectorizer
        corpus_index: se asume que el índice del corpus es la clave y que los vectores están ordenados igual como lo están los docuemntos de corpus, 
            de forma que se puede asignar clave de documento a clave de vector_documento cruzando por posición
        query_index: se asume que el índice de las queries es la clave y que los vectores están ordenados igual como lo están las queries, 
            de forma que se puede asignar clave de query a clave de vector_query cruzando por posición
        rank: se retornan los rank productos más relevantes
    
    Retorna:
        un pd.DataFrame con una columna por query conteniendo los product_id retornados ordenados por relevancia (de mayor a menor score - de menor a mayor distancia coseno -)
        los nombres de coluna son los query_id
    """
    scores = vectorizer.score(queries).T # BM25Vectorizer retorna los rankings como filas; se traspone por coherencia con run_query_dense, que los retorna en columnas 
    sparse_rank = np.argsort(-scores, axis=0)[:k,:] # se se niega scores, de forma que el orden es descendente
    sparse_rank_df = pd.DataFrame(sparse_rank).apply(lambda x: corpus_index[x])
    sparse_rank_df.columns = queries.index.to_list()
    sparse_rank_df.attrs['name'] = name
    return sparse_rank_df

def evaluate_compare_queries(
    metrics: Literal['recall', 'nDCG', 'MRR'],
    rank_df_list: Sequence[pd.DataFrame],
    labeled_query_results_sorted: pd.DataFrame,
    valid_esci_labels: list,
) -> pd.DataFrame:
    """
    genera un tabla comparativa por la métricas indicadas de un mismo conjunto de queries aplicadas a distintos tipos de codificación
    los rankings a comparar deben habrer sido obtenidos con run_query_sparse o run_query_dense

    Args:
        - metrics: las metricas que se quieren calcular; pueden ser recall@k, nDGC@k, MRR@k, siendo k igual al número de resultados de los rankings a comparar
        - rank_df_list: una lista de rankings tal como es devuelto por la función run_query_sparse o la función run_query_dense, todos para un mismo set de queries y una misma k
        - labeled_query_results_sorted: un pd.DataFrame de de resultados etiquetados a las queries de prueba con las que se han obtenido los rangos, tal como se espera en las funciones de métricas, con los productos ordenados de mayor a menor relevancia
        - valid_esci_labels: esci_labels que se consideran en la evaluación
    """
    metric_functions={'recall': recall_at_k, 'nDCG': nDCG_at_k, 'MRR': MRR_at_k}
    if not rank_df_list:
        raise ValueError("rank_df_list must contain at least one ranking DataFrame")

    k = len(rank_df_list[0])
    query_ids = rank_df_list[0].columns.to_list()
    for position, rank_df in enumerate(rank_df_list, start=1):
        if len(rank_df) != k:
            raise ValueError("all ranking DataFrames must have the same number of rows (k)")
        if rank_df.columns.to_list() != query_ids:
            raise ValueError("all ranking DataFrames must contain the same query columns in the same order")
    
    # DataFrames have no __name__.  Callers can optionally provide a readable
    # label through ``rank_df.attrs['name']``; otherwise use stable defaults.
    rank_names = [rank_df.attrs.get("name", f"rank_{position}")
             for position, rank_df in enumerate(rank_df_list, start=1)]
    if len(set(rank_names)) != len(rank_names):
        raise ValueError("ranking DataFrame names must be unique")
    metric_names = [f'{metric_name}@{k}' for metric_name in metrics]

    columns = pd.MultiIndex.from_product([metric_names, rank_names])
    result = pd.DataFrame(index=query_ids, columns=columns, dtype=float)
    for metric, metric_name in zip(metrics, metric_names):
        for query_id in query_ids:
            for rank_name, rank_df in zip(rank_names, rank_df_list):
                result.loc[query_id, (metric_name, rank_name)] = metric_functions[metric](
                    query_id, valid_esci_labels, rank_df, labeled_query_results_sorted, k
                )
    return result


#### BASE DE DATOS VECTORIAL, CHROMADB

def _to_plain_data(value: Any) -> Any:
    """Convert Chroma/Pydantic values into regular Python containers."""
    if hasattr(value, "model_dump"):
        try:
            return _to_plain_data(value.model_dump(mode="json"))
        except TypeError:  # Pydantic versions without the ``mode`` argument.
            return _to_plain_data(value.model_dump())
    if hasattr(value, "dict"):
        return _to_plain_data(value.dict())
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    return value

def get_collection_schema_and_configuration(client: Any, collection_name: str) -> dict[str, Any]:
    """Return a collection's schema and configuration from a Chroma client.

    Args:
        client: A Chroma client, for example ``chromadb.PersistentClient``.
        collection_name: Name of an existing Chroma collection.

    Returns:
        A serialisable dictionary with the collection identity, metadata, schema,
        and configuration. Schema and configuration are returned as plain Python
        dictionaries/lists, so they can be displayed or encoded as JSON.

    Raises:
        ValueError: If the installed Chroma version does not expose the requested
            collection properties.
    """
    collection = client.get_collection(name=collection_name)
    missing = [
        property_name
        for property_name in ("schema", "configuration")
        if not hasattr(collection, property_name)
    ]
    if missing:
        raise ValueError(
            "This Chroma version does not expose collection "
            f"{', '.join(missing)}. Upgrade chromadb to retrieve it."
        )

    return {
        "id": collection.id,
        "name": collection.name,
        "metadata": _to_plain_data(collection.metadata),
        "schema": _to_plain_data(collection.schema),
        "configuration": _to_plain_data(collection.configuration),
    }


def db_ingestion(collection: chromadb.Collection, file: str, batch_size: int =1500):
    """
    - Importa un catálogo a una colección
    - El campo 'text' se importa como document
    - los embeddings se generan con embed_documents_ge2 (modelo gemini-embedding-2) con dimensionalidad 768. El texto para los embeddings se procesa con parse_corpus
    - Toma el campo 'record_id' como clave (campo 'ids' cromadb). 
    - cromadb ignora los datos añadidos con un ids ya existente. Aun así se filtran fuera en la función los record_id que ya están en base de datos para evitar procesarlos innecesariamente en parse_corpus y embed_documents_ge2
    - Como metadatos se importan: 
        - 'brand': para filtrado 
        - 'title': para filtrado 
        - 'product_id': para trazabilidad
        - 'active': permite disponer de productos o versiones de productos descatalogados, lo que evita tener que rehacer el grafo de HNSW ante modificaciones; simplemente antes de retornar resultados se filtran fuera los inactivos

    Args:
        collection: una colección de chromadb
        file: un csv conteniendo un catálogo de tipo 'version 1' y en castellano ('locale' es)
        batch_size: la importación se hacde por lotes; batch_size indica el tamaño de los lotes

    Returns:
        número de ids distintos presentes en la base de datos al final de la ingesta
    """
    ids = collection.get(include=[])['ids']
    with pd.read_csv(os.path.join(project_root, 'datos', file), sep=',', decimal='.', encoding='utf-8', chunksize=batch_size) as reader:
        for chunk_df in reader:
            proc_chunk_df = chunk_df[~chunk_df['record_id'].isin(ids)]
            if proc_chunk_df.empty:
                continue
            proc_chunk_df.set_index('product_id', inplace=True)
            parsed_chunk = parse_corpus(proc_chunk_df)
            embeddings = embed_documents_ge2(parsed_chunk['string2embed'], 'RETRIEVAL_DOCUMENT', output_dimensionality=768)
            metadata_columns = ['brand', 'title', 'product_id', 'active']
            metadatas = (
                proc_chunk_df.reset_index()[metadata_columns]
                .to_dict(orient="records")
            )
            collection.add(
                ids=proc_chunk_df['record_id'].astype(str).to_list(), # ids debe ser un string
                embeddings=embeddings,
                documents=proc_chunk_df['text'].to_list(),
                metadatas=metadatas,
            ) 
    return collection.count()


# en chromadb tanto el inner product (ip) como el cosine similarity (cosine) se dan como 'distancia' devolviendo (1 - x) (ip asume vectores normalizados)
def db_query(collection: chromadb.Collection, query_texts: list[str] | pd.Series = None, brands: list[str] = None, top_k:int = 10):
    """
    ejecuta una lista de queries en la base de datos

    Args:
        - collection: la colección sobre la que actuar
        - query_texts: lista o pd.Series con el listado de queries (si solamente hay una, un listado con una sola). opcional (por defecto None)
        - brand: marca por la que filtrar, opcional (por defecto None)
        - top_k: retornará los top_k resultados más relevantes

    Returns:
        - si no se especifica ni query_texts ni brand, no retorna nada 
        - siempre filtra por registros con active a True
        - si se indica query_text, retorna un dataframe con los top_k resultados más relevantes para cada query y conteniendo:
            - índices de query_text (si se pasó como una lista son enteros no negativos empezando por 0)
            - product_id
            - score (similitud coseno)
            - title
            - document
        - si no se indica query_text y solamente se indica el filtro por marca, lo mismo, pero sin el campo score.
        - excepciones:
            - si no hay productos de la marca especificada en la base de datos devuelve f'no products for brand {brand}'
            - si con query_text no se encuentran resultados, retorna 'no products found for the query'
            - si la colección está vacía retorna 'empty collection'
        
    """
    if not collection.count():
        return 'empty collection'

    if brands:
        # chequeo de si hay productos registrados con esa marca
        if [[]] == collection.get(where={'$and': [{'brand': {'$in': brands}}, {'active': True}]}, include=[])['ids']:  # returna solanete los ids            
            return f'no products for supplied brands'

        # caso en que en la consulta solamente haya el fitro por marca
        if not(isinstance(query_texts, pd.Series) or isinstance(query_texts, list)):
            res = collection.get(where={'$and': [{'brand': {'$in': brands}}, {'active': True}]}, # se filtra siempre por registros activos
                                 n_results = top_k,
                                 include=['metadatas', 'documents'])
            D={'query_id':[],
               'product_id':[],
               'brand':[],
               'title':[],
               'document':[]}
            for query_id, metadatas, documents in zip(query_texts.index, res['metadatas'], res['documents']):
                for metadata, document in zip(metadatas, documents):
                    D['query_id'].append(query_id)
                    D['product_id'].append(metadata['product_id'])
                    D['brand'].append(metadata['brand'])
                    D['title'].append(metadata['title'])
                    D['document'].append(document)
            return pd.DataFrame(D)
    
    if isinstance(query_texts, list):
        query_texts = pd.Series(query_texts)
    if isinstance(query_texts, pd.Series):
        if isinstance(query_texts, list):
            query_texts = pd.Series(query_texts)
        query_parsed = query_texts.apply(text_parser)
        query_embeddings = embed_documents_ge2(query_parsed, 'RETRIEVAL_QUERY', 768)
        if brands: # query con query_text y filtro por brand
            res = collection.query(query_embeddings = query_embeddings,
                                where={'$and': [{'brand': {'$in': brands}}, {'active': True}]},
                                n_results = top_k,
                                include=['metadatas', 'distances', 'documents'])
        else: # query con query_text pero sin filtro por marca
            res = collection.query(query_embeddings = query_embeddings,
                                where={'active': True},
                                n_results = top_k,
                                include=['metadatas', 'distances', 'documents'])
        if res['ids'] == [[]]:
            return 'no products found for the query'
        D={'query_id':[],
        'product_id':[],
        'score':[],
        'brand':[],
        'title':[],
        'document':[]}
        for query_id, metadatas, distances, documents in zip(query_texts.index, res['metadatas'], res['distances'], res['documents']):
            for metadata, distance, document in zip(metadatas, distances, documents):
                D['query_id'].append(query_id)
                D['product_id'].append(metadata['product_id'])
                D['score'].append(1-distance)
                D['brand'].append(metadata['brand'])
                D['title'].append(metadata['title'])
                D['document'].append(document)
        return pd.DataFrame(D)


def db_modify(collection: chromadb.Collection,  records: pd.DataFrame, dup_thresh: int = 0.81, force_addition: bool = False):
    """
    - modifica registros en la bae de datos, detectando posibles duplicados
    - dado que el índice es de tipo HNSW, eleminar registros podría alterar el grafo. Es es por ello que no se eliminan registros, 
    solamente se marcan cambiando el metadata 'active' a False, Sería una tarea de administrdor supervisar cuando el número de registros
    inactivos se considere lo suficientemente elevado como para proceder a eliminarlos y reahcer el índice 
    
    Args:
        - collection: la colección sobre la que actuar
        - registers: un pd.DataFrame que debe contener las siguientes columnas:
                - operation: puede valer DELETE o UPSERT
                - record_id: mismo formato que los ids de la base de datos; si existe el contenido se modifica, sino se añade registro. deben ser únicos
                - product_id
                - title
                - document
        - dup_thresh: antes de llevar a cabo una adición se realiza una query con 'text' y 'brand' del record. Si el scroe supera dup_thresh, se marca como registro sospechoso de ser un duplicado
        - force_addition: si está a False (valor por defecto) no añade los registros sospechosos de ser duplicados. en otro caso, sí
    
    Returns:
        - si hay record_id duplicados cancela la ejecución y retorna cuales son
        - un pd.Series, con índice record_id, con con información sobre lo hecho para cada registro: 
            - deleted, 
            - updated, 
            - added, 
            - duplicate_suspect_not_added, 
            - duplicate_suspect_added, 
            - none (caso update o delete sobre registro no existente). 
    """

    rvc = records['record_id'].value_counts()
    duplicated_records = rvc[rvc.apply(lambda x: x >1)]
    if not duplicated_records.empty:
        return pd.DataFrame(duplicated_records, columns=['duplicated record_id'])
    
    existing_records =  collection.get(ids=records['record_id'].to_list(), include=[])['ids']
    deleted_records = collection.get(where={'active': False}, include=[])['ids']

    records.sort_values(by=['sequence'])
    #records.set_index('record_id', inplace=True)
    actions = pd.Series(['none']*len(records), index=records['record_id'])
    corpus = parse_corpus(records)
    #metadata_columns = ['brand', 'title', 'product_id']

    # la operación sería más eficiente en batches por tipo de operación, pero para seguir el enunciado y respectar el orden en 'sequence', se itera en ese orden
    # chromadb registra los intentos de update a registros que no existen y upsert automáticamente ejecuta una adición o un update, según correcponda, pero para 
    # segir el enunciado se hace el 'parse' explícito
    for i, row in corpus.iterrows():
        if row['record_id'] in existing_records and row['record_id'] not in deleted_records:
            if row['operation'] == 'UPSERT':
                
                collection.update(
                    ids=[row['record_id']],
                    embeddings = embed_documents_ge2(documents=row['string2embed'],
                                                     task_type='RETRIEVAL_DOCUMENT',
                                                     output_dimensionality=768),
                    metadatas=[{'brand': row['brand'], 'title': row['title'], 'product_id': row['product_id'], 'active': True}],
                    documents=[row['text']]
                )
                actions.loc[row['record_id']] = 'updated'
            if row['operation'] == 'DELETE': # se leen los metadatos y se reescriben con metadata['active'] = False
                metadata = collection.get(ids=[row['record_id']], include=['metadatas'])['metadatas'][0]
                metadata['active'] = False
                collection.update(
                    ids=[row['record_id']],
                    metadatas=[metadata],
                )
                actions.loc[row['record_id']] = 'deleted'
        else:
            if row['operation'] == 'UPSERT':
                res = db_query(collection, query_texts=[row['text']], brands=[row['brand']], top_k=1)
                score = 0
                if isinstance(res, pd.DataFrame):
                    score = res.loc[0,'score']
                if force_addition or score < dup_thresh:
                    collection.add(
                        ids=[row['record_id']],
                        embeddings = embed_documents_ge2(documents=row['string2embed'],
                                                            task_type='RETRIEVAL_DOCUMENT',
                                                            output_dimensionality=768),
                        metadatas=[{'brand': row['brand'], 'title': row['title'], 'product_id': row['product_id'], 'active': True}],
                        documents=[row['text']],
                    )
                    actions.loc[row['record_id']] = 'added' if score < dup_thresh else 'duplicate_suspect_added'
                else:
                    actions.loc[row['record_id']] = 'duplicate_suspect_not_added'
    return actions
                    


                    





    









    
