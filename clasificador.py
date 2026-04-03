import os
import re
import unicodedata
from collections import Counter
import requests
import torch
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig
from sentence_transformers import SentenceTransformer
import faiss

GENEROS       = {0:'Novela', 1:'Cuento', 2:'Poesía', 3:'Ensayo', 4:'Teatro', 5:'Fábula', 6:'Crónica'}
NUM_LABELS    = 7
TIPOS_LECTURA = {0:'Infantil', 1:'Juvenil', 2:'Académica', 3:'Entretenimiento'}
NUM_TIPOS     = 4

# Ruta local — evita descargar de HuggingFace cada vez. Nicaury Díaz 23-SISN-2-028
BASE_DIR     = os.path.dirname(__file__)
MODEL_LOCAL  = os.path.join(BASE_DIR, 'bert_local')
MODEL_NAME   = MODEL_LOCAL if os.path.exists(MODEL_LOCAL) else 'bert-base-multilingual-cased'

RUTA_GENERO  = os.path.join(BASE_DIR, 'modelo_genero.pt')
RUTA_TIPO    = os.path.join(BASE_DIR, 'modelo_tipo.pt')

_tokenizer     = None
_modelo_genero = None
_modelo_tipo   = None
_embedder      = None
_autores_index = None
_autores_meta  = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# CARGA DE MODELOS
def _construir_modelo(ruta_pt: str, num_labels: int):
    checkpoint = torch.load(ruta_pt, map_location=device)
    config     = BertConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    modelo     = BertForSequenceClassification(config)

    tiene_classifier = any('classifier' in k for k in checkpoint.keys())
    if tiene_classifier:
        modelo.load_state_dict(checkpoint)
    else:
        estado = modelo.state_dict()
        estado.update({k: v for k, v in checkpoint.items() if k in estado})
        modelo.load_state_dict(estado)

    return modelo.to(device).eval()


def _cargar_modelos():
    global _tokenizer, _modelo_genero, _modelo_tipo, _embedder
    global _autores_index, _autores_meta

    if _tokenizer is not None:
        return

    _tokenizer     = AutoTokenizer.from_pretrained(MODEL_NAME)
    _modelo_genero = _construir_modelo(RUTA_GENERO, NUM_LABELS) if os.path.exists(RUTA_GENERO) else None
    _modelo_tipo   = _construir_modelo(RUTA_TIPO,   NUM_TIPOS)  if os.path.exists(RUTA_TIPO)   else None

    try:
        _embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception:
        _embedder = None

    faiss_path = os.path.join(BASE_DIR, 'autor_index.faiss')
    meta_path  = os.path.join(BASE_DIR, 'autor_meta.npy')
    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        _autores_index = faiss.read_index(faiss_path)
        _autores_meta  = np.load(meta_path, allow_pickle=True).tolist()


# PREDICCIONES Nicaury Diaz 23-SISN-2-028

def _predecir_genero(texto: str) -> dict:
    if _modelo_genero is None:
        return {'genero': 'N/D', 'confianza': 0.0, 'todas': {g: 0.0 for g in GENEROS.values()}}
    enc = _tokenizer(texto, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        logits = _modelo_genero(
            input_ids=enc['input_ids'].to(device),
            attention_mask=enc['attention_mask'].to(device)
        ).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred  = probs.argmax().item()
    return {
        'genero':    GENEROS[pred],
        'confianza': probs[pred].item(),
        'todas':     {GENEROS[i]: round(probs[i].item(), 4) for i in range(NUM_LABELS)},
    }


def _predecir_tipo(texto: str) -> dict:
    if _modelo_tipo is None:
        return {'tipo': 'N/D', 'confianza': 0.0, 'todas': {t: 0.0 for t in TIPOS_LECTURA.values()}}
    enc = _tokenizer(texto, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        logits = _modelo_tipo(
            input_ids=enc['input_ids'].to(device),
            attention_mask=enc['attention_mask'].to(device)
        ).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred  = probs.argmax().item()
    return {
        'tipo':      TIPOS_LECTURA[pred],
        'confianza': probs[pred].item(),
        'todas':     {TIPOS_LECTURA[i]: round(probs[i].item(), 4) for i in range(NUM_TIPOS)},
    }


# Patrones comunes en encabezados de libros/documentos
_PATRONES_AUTOR = [
    # "Autor: Juan Pérez" / "Author: Jeime Freiser"
    r'(?:autor(?:a)?|author)\s*[:\-–]\s*([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+){1,4})',
    # "Por Juan Pérez" / "By Jeime Fraiser"
    r'(?:por|by)\s+([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+){1,4})',
    # "© 2001 Juan Pérez"
    r'©\s*\d{4}\s+([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+){1,4})',
    # Línea sola que parece nombre propio (en las primeras líneas)
    r'^([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+){1,3})\s*$',
]

def _extraer_autor_del_texto(texto: str) -> str | None:
    zona_busqueda = texto[:1000] + '\n' + texto[-500:]

    for patron in _PATRONES_AUTOR:
        flags = re.IGNORECASE | re.MULTILINE
        coincidencias = re.findall(patron, zona_busqueda, flags=flags)
        for nombre in coincidencias:
            nombre = nombre.strip()
            palabras = nombre.split()
            if len(palabras) < 2:
                continue
            stopwords = {'capítulo', 'chapter', 'título', 'title', 'prólogo',
                         'introducción', 'prefacio', 'editorial', 'primera', 'segunda'}
            if any(p.lower() in stopwords for p in palabras):
                continue
            return nombre

    return None


def _invertir_nombre_gutenberg(nombre: str) -> str:
    nombre = re.sub(r',?\s*\d{4}-\d{4}', '', nombre).strip()
    if ',' not in nombre:
        return nombre
    partes    = [p.strip() for p in nombre.split(',', 1)]
    apellido  = partes[0]
    prenombre = partes[1] if len(partes) > 1 else ''
    return f'{prenombre} {apellido}' if prenombre else apellido


def _extraer_frases_clave(texto: str, n: int = 3) -> list[str]:
    oraciones = re.split(r'(?<=[.!?¿¡])\s+', texto.strip())
    candidatas = [o.strip() for o in oraciones if 40 < len(o.strip()) < 250]

    if not candidatas:
        # Fallback: trozos uniformes del texto
        paso = max(1, len(texto) // n)
        return [texto[i*paso:(i+1)*paso].strip()[:150] for i in range(n)]
    indices = [int(len(candidatas) * f) for f in [0.1, 0.5, 0.85]]
    return [candidatas[min(i, len(candidatas)-1)][:200] for i in indices]


def _titulo_desde_texto(texto: str) -> str | None:
    lineas = texto[:500].split('\n')
    for linea in lineas:
        linea = linea.strip()
        if 3 <= len(linea.split()) <= 10:
            if linea.isupper() or re.match(r'^([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]*\s*)+$', linea):
                return linea
    return None


# BÚSQUEDAS EN APIS 

def _buscar_gutenberg(queries: list[str]) -> list[dict]:
    candidatos = []
    for query in queries:
        try:
            r = requests.get(
                'https://gutendex.com/books/',
                params={'search': query[:100], 'languages': 'en,es'},
                timeout=10
            )
            for libro in r.json().get('results', [])[:5]:
                autores = libro.get('authors', [])
                if autores:
                    nombre = _invertir_nombre_gutenberg(autores[0].get('name', ''))
                    if nombre:
                        candidatos.append({
                            'autor':  nombre,
                            'titulo': libro.get('title', 'N/D'),
                            'fuente': 'Gutenberg',
                            'score':  libro.get('download_count', 0) + 1,
                        })
        except Exception:
            pass
    return candidatos


def _buscar_open_library(queries: list[str]) -> list[dict]:
    candidatos = []
    for query in queries:
        try:
            r = requests.get(
                'https://openlibrary.org/search.json',
                params={'q': query, 'limit': 5,
                        'fields': 'title,author_name,first_publish_year'},
                timeout=10
            )
            for doc in r.json().get('docs', [])[:5]:
                autores = doc.get('author_name', [])
                if autores:
                    candidatos.append({
                        'autor':  autores[0],
                        'titulo': doc.get('title', 'N/D'),
                        'año':    doc.get('first_publish_year', 'N/D'),
                        'fuente': 'OpenLibrary',
                        'score':  1,
                    })
        except Exception:
            pass
    return candidatos


def _buscar_google_books(queries: list[str]) -> list[dict]:
    candidatos = []
    for query in queries:
        try:
            r = requests.get(
                'https://www.googleapis.com/books/v1/volumes',
                params={'q': query[:100], 'maxResults': 5, 'printType': 'books'},
                timeout=10
            )
            for item in r.json().get('items', [])[:5]:
                info    = item.get('volumeInfo', {})
                autores = info.get('authors', [])
                if autores:
                    candidatos.append({
                        'autor':  autores[0],
                        'titulo': info.get('title', 'N/D'),
                        'fuente': 'GoogleBooks',
                        'score':  1,
                    })
            if candidatos:
                break  
        except Exception:
            pass
    return candidatos


# CONSENSO ENTRE CANDIDATOS DE AUTOR PARA DETERMINAR EL MÁS PROBABLE

def _normalizar_nombre(nombre: str) -> str:
    nfkd = unicodedata.normalize('NFKD', nombre)
    sin_tildes = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', sin_tildes).strip().lower()


def _consenso_autor(candidatos: list[dict]) -> tuple[str, str]:
    grupos: dict[str, dict] = {}
    for c in candidatos:
        clave = _normalizar_nombre(c['autor'])
        if clave not in grupos:
            grupos[clave] = {'nombres': [], 'score_total': 0, 'fuentes': set()}
        grupos[clave]['nombres'].append(c['autor'])
        grupos[clave]['score_total'] += c.get('score', 1)
        grupos[clave]['fuentes'].add(c.get('fuente', '?'))

    if not grupos:
        return 'No identificado', 'baja'

    mejor_clave = max(
        grupos,
        key=lambda k: (len(grupos[k]['fuentes']), grupos[k]['score_total'])
    )
    info = grupos[mejor_clave]
    nombre_original = Counter(info['nombres']).most_common(1)[0][0]

    n_fuentes = len(info['fuentes'])
    nivel = 'alta' if n_fuentes >= 3 else 'media' if n_fuentes >= 2 else 'baja'
    return nombre_original, nivel


def _predecir_autor(texto: str) -> dict:
    autor_directo = _extraer_autor_del_texto(texto)
    if autor_directo:
        return {'autor': autor_directo, 'confianza': 'alta (texto)'}
    frases     = _extraer_frases_clave(texto, n=3)
    titulo     = _titulo_desde_texto(texto)
    queries    = frases.copy()
    if titulo:
        queries.insert(0, titulo)  
    candidatos = []
    candidatos.extend(_buscar_gutenberg(queries[:2]))
    candidatos.extend(_buscar_open_library(queries[:2]))
    candidatos.extend(_buscar_google_books(queries[:2]))

    if not candidatos:
        return {'autor': 'No identificado', 'confianza': 'N/D'}

    autor, nivel = _consenso_autor(candidatos)
    return {'autor': autor, 'confianza': nivel}


# ENTRADA PÚBLICA Nicaury Diaz 23-SISN-2-028
def clasificar_texto(texto: str) -> dict:
    _cargar_modelos()
    texto = texto.strip()
    if not texto:
        return {'error': 'Texto vacío'}
    try:
        rg = _predecir_genero(texto)
        rt = _predecir_tipo(texto)
        ra = _predecir_autor(texto)
        return {
            'genero':        rg['genero'],
            'conf_genero':   f"{rg['confianza']*100:.1f} %",
            'barras_genero': {k: round(v*100, 1) for k, v in rg['todas'].items()},
            'tipo':          rt['tipo'],
            'conf_tipo':     f"{rt['confianza']*100:.1f} %",
            'barras_tipo':   {k: round(v*100, 1) for k, v in rt['todas'].items()},
            'autor':         ra['autor'],
            'conf_autor':    ra['confianza'],
            'error':         None,
        }
    except Exception as exc:
        return {'error': str(exc)}


def resultado_texto(resultado: dict) -> str:
    if resultado.get('error'):
        return f"❌ Error en clasificación: {resultado['error']}"
    lineas = [
        "🌸 LETROVA — CLASIFICACIÓN LITERARIA",
        "",
        f"📖 Género: {resultado['genero']} ({resultado['conf_genero']})",
    ]
    for nombre, pct in resultado['barras_genero'].items():
        lineas.append(f"   {nombre}: {pct:.1f}%")
    lineas += [
        "",
        f"🎯 Tipo: {resultado['tipo']} ({resultado['conf_tipo']})"
    ]
    for nombre, pct in resultado['barras_tipo'].items():
        lineas.append(f"   {nombre}: {pct:.1f}%")
    lineas += [
        "",
        f"✍️ Autor: {resultado['autor']} ({resultado['conf_autor']})"
    ]
    return "\n".join(lineas)