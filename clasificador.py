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
    # Captura: "H. G. Wells", "J. K. Rowling", "R. L. Stevenson", etc.
    r'\b([A-ZÁÉÍÓÚÑÜ]\.(?:\s+[A-ZÁÉÍÓÚÑÜ]\.)+\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]{2,}(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+)*)\b',

    #  nombres con iniciales precedidos de "by" o "por"
    r'(?:por|by)\s+([A-ZÁÉÍÓÚÑÜ]\.(?:\s+[A-ZÁÉÍÓÚÑÜ]\.)*\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+)*)',

    #línea sola con iniciales en encabezado 
    r'^([A-ZÁÉÍÓÚÑÜ]\.(?:\s+[A-ZÁÉÍÓÚÑÜ]\.)*\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+)*)\s*$',

    # Con etiqueta explícita "Autor: Gabriel García Márquez", "Author: Isabel Allende", etc.  
    r'(?:autor(?:a)?|author)\s*[:\-–]\s*([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]*\.?(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]*\.?){1,4})',
    r'(?:por|by)\s+([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]*\.?(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]*\.?){1,4})',
    r'©\s*\d{4}\s+([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]*\.?(?:\s+[A-ZÁÉÍÓÚÑÜ][a záéíóúñü]*\.?){1,4})',

    # Todo mayúsculas con etiqueta
    r'(?:autor(?:a)?|author|por|by)\s*[:\-–]?\s*([A-ZÁÉÍÓÚÑÜ]{2,}(?:\s+[A-ZÁÉÍÓÚÑÜ\.]{2,}){1,4})',

    # Línea sola 
    r'^([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]*\.?){1,3})\s*$',
]

# Palabras que NUNCA son parte de un nombre de autor
_STOPWORDS_AUTOR = {
    'the', 'and', 'with', 'from', 'into', 'upon', 'over', 'that', 'this',
    'were', 'have', 'been', 'blue', 'frock', 'coat', 'shop', 'window',
    'door', 'tion', 'sion', 'ment', 'of', 'in', 'on', 'at', 'to', 'a',
    'capítulo', 'chapter', 'título', 'title', 'prólogo', 'introducción',
    'prefacio', 'editorial', 'primera', 'segunda', 'section', 'direction',
}

def _es_nombre_valido(nombre: str) -> bool:
    palabras = nombre.split()
    
   
    if len(palabras) < 2:
        return False
    
    
    if any(p.lower() in _STOPWORDS_AUTOR for p in palabras):
        return False

    if any(re.search(r'(tion|sion|ment|ness|ity)$', p, re.I) for p in palabras):
        return False

    for p in palabras:
        es_inicial = bool(re.match(r'^[A-ZÁÉÍÓÚÑÜ]\.$', p))
        es_palabra_normal = bool(re.match(r'^[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+\.?$', p))
        if not (es_inicial or es_palabra_normal):
            return False
        
    if any(len(p) > 20 for p in palabras):
        return False

    tiene_apellido = any(
        re.match(r'^[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]{2,}$', p) for p in palabras
    )
    return tiene_apellido


def _extraer_autor_del_texto(texto: str) -> str | None:
    texto_norm = texto.replace('\r\n', '\n').replace('\r', '\n')
    encabezado = texto_norm[:600]
    m = re.search(
        r'\bby\s+([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+){1,3})',
        encabezado, re.IGNORECASE
    )
    if m:
        nombre = m.group(1).strip()
        if _es_nombre_valido(nombre):
            return nombre

    zona_etiquetada = texto_norm[:3000] + '\n' + texto_norm[-500:]
    zona_encabezado = texto_norm[:600]   

    for i, patron in enumerate(_PATRONES_AUTOR):
        if i in (0, 1, 2, 7):
            zona = zona_encabezado
        else:
            zona = zona_etiquetada

        coincidencias = re.findall(patron, zona, flags=re.IGNORECASE | re.MULTILINE)

        for nombre in coincidencias:
            nombre = nombre.strip()
            if nombre.isupper():
                nombre = nombre.title()

            if _es_nombre_valido(nombre):
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


# ENRIQUECIMIENTO DEL AUTOR: nacionalidad, época y obras representativas

_GENTILICIOS = {
    'español': 'España', 'española': 'España',
    'colombiano': 'Colombia', 'colombiana': 'Colombia',
    'argentino': 'Argentina', 'argentina': 'Argentina',
    'mexicano': 'México', 'mexicana': 'México',
    'chileno': 'Chile', 'chilena': 'Chile',
    'peruano': 'Perú', 'peruana': 'Perú',
    'dominicano': 'República Dominicana', 'dominicana': 'República Dominicana',
    'cubano': 'Cuba', 'cubana': 'Cuba',
    'venezolano': 'Venezuela', 'venezolana': 'Venezuela',
    'uruguayo': 'Uruguay', 'uruguaya': 'Uruguay',
    'paraguayo': 'Paraguay', 'paraguaya': 'Paraguay',
    'boliviano': 'Bolivia', 'boliviana': 'Bolivia',
    'ecuatoriano': 'Ecuador', 'ecuatoriana': 'Ecuador',
    'panameño': 'Panamá', 'panameña': 'Panamá',
    'costarricense': 'Costa Rica',
    'guatemalteco': 'Guatemala', 'guatemalteca': 'Guatemala',
    'hondureño': 'Honduras', 'hondureña': 'Honduras',
    'nicaragüense': 'Nicaragua',
    'salvadoreño': 'El Salvador', 'salvadoreña': 'El Salvador',
    'puertorriqueño': 'Puerto Rico', 'puertorriqueña': 'Puerto Rico',
    'francés': 'Francia', 'francesa': 'Francia',
    'italiano': 'Italia', 'italiana': 'Italia',
    'portugués': 'Portugal', 'portuguesa': 'Portugal',
    'alemán': 'Alemania', 'alemana': 'Alemania',
    'ruso': 'Rusia', 'rusa': 'Rusia',
    'británico': 'Reino Unido', 'británica': 'Reino Unido',
    'estadounidense': 'Estados Unidos',
    'norteamericano': 'Estados Unidos', 'norteamericana': 'Estados Unidos',
    'spanish': 'España', 'colombian': 'Colombia', 'argentine': 'Argentina',
    'mexican': 'México', 'chilean': 'Chile', 'peruvian': 'Perú',
    'dominican': 'República Dominicana', 'cuban': 'Cuba',
    'venezuelan': 'Venezuela', 'uruguayan': 'Uruguay',
    'french': 'Francia', 'italian': 'Italia', 'portuguese': 'Portugal',
    'german': 'Alemania', 'russian': 'Rusia', 'british': 'Reino Unido',
    'american': 'Estados Unidos',
    'english': 'Reino Unido', 'welsh': 'Reino Unido', 'scottish': 'Reino Unido',
}

def _enriquecer_autor(nombre_autor: str) -> dict:
    info = {
        'nacionalidad': 'N/D',
        'epoca':        'N/D',
        'obras':        [],
    }

    # 1. OpenLibrary: época y obras
    try:
        r_busqueda = requests.get(
            'https://openlibrary.org/search/authors.json',
            params={'q': nombre_autor},
            timeout=10
        )
        docs = r_busqueda.json().get('docs', [])
        apellido_buscado = _normalizar_nombre(nombre_autor.split()[-1])
        doc = None
        for candidato in docs[:5]:
            nombre_doc = _normalizar_nombre(candidato.get('name', ''))
            if apellido_buscado in nombre_doc and candidato.get('birth_date'):
                doc = candidato
                break
        if doc is None:
            for candidato in docs[:5]:
                if candidato.get('birth_date'):
                    doc = candidato
                    break
        if doc is None and docs:
            doc = docs[0]

        if doc:
            autor_key  = doc.get('key', '')
            birth_date = doc.get('birth_date', '')
            death_date = doc.get('death_date', '')

            if birth_date:
                anio_nac = re.search(r'\d{4}', birth_date)
                anio_mue = re.search(r'\d{4}', death_date) if death_date else None
                if anio_nac and anio_mue:
                    info['epoca'] = f"{anio_nac.group()}–{anio_mue.group()}"
                elif anio_nac:
                    info['epoca'] = f"nacido en {anio_nac.group()}"

            if autor_key:
                r_obras = requests.get(
                    f'https://openlibrary.org/authors/{autor_key}/works.json',
                    params={'limit': 10},
                    timeout=10
                )
                titulos_vistos: set[str] = set()
                for entrada in r_obras.json().get('entries', []):
                    titulo = entrada.get('title', '').strip()
                    if titulo and titulo.lower() not in titulos_vistos:
                        titulos_vistos.add(titulo.lower())
                        info['obras'].append(titulo)
                    if len(info['obras']) >= 6:
                        break
    except Exception:
        pass

    # 2. Wikipedia (es/en): nacionalidad y época si aún falta
    partes_nombre = nombre_autor.split()
    variantes_wiki = [nombre_autor.replace(' ', '_')]
    if len(partes_nombre) == 3:
        variantes_wiki.append(f"{partes_nombre[0]}_{partes_nombre[-1]}")

    for nombre_wiki in variantes_wiki:
        for lang in ('es', 'en'):
            try:
                r_wiki = requests.get(
                    f'https://{lang}.wikipedia.org/api/rest_v1/page/summary/{nombre_wiki}',
                    headers={'User-Agent': 'LetrOva/1.0 (clasificador literario; nicaury@ejemplo.com)'},
                    timeout=10
                )
                if r_wiki.status_code != 200:
                    continue
                extracto_raw = r_wiki.json().get('extract', '')
                extracto = unicodedata.normalize('NFKC', extracto_raw)
                apellido = partes_nombre[-1].lower()
                if apellido not in extracto.lower():
                    continue

                if info['nacionalidad'] == 'N/D':
                    for gentilicio, pais in _GENTILICIOS.items():
                        if re.search(rf'\b{re.escape(gentilicio)}\b', extracto, re.IGNORECASE):
                            info['nacionalidad'] = pais
                            break

                if info['epoca'] == 'N/D':
                    m = re.search(r'\((\d{4})\s*[–\-]\s*(\d{4})\)', extracto)
                    if m:
                        info['epoca'] = f"{m.group(1)}–{m.group(2)}"
                    else:
                        m_largo = re.search(
                            r'\([^)]*?(\d{4})\s*[–\-]\s*[^)]*?(\d{4})\)',
                            extracto
                        )
                        if m_largo:
                            info['epoca'] = f"{m_largo.group(1)}–{m_largo.group(2)}"
                        else:
                            m_born = re.search(r'(?:born|nacido|nació)[^\d]*(\d{4})', extracto, re.IGNORECASE)
                            m_died = re.search(r'(?:died|murió|fallecido)[^\d]*(\d{4})', extracto, re.IGNORECASE)
                            if m_born and m_died:
                                info['epoca'] = f"{m_born.group(1)}–{m_died.group(1)}"
                            elif m_born:
                                info['epoca'] = f"nacido en {m_born.group(1)}"
                            else:
                                m2 = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', extracto)
                                if m2:
                                    info['epoca'] = f"circa {m2.group(1)}"

                if info['nacionalidad'] != 'N/D' and info['epoca'] != 'N/D':
                    break
            except Exception:
                pass
        if info['nacionalidad'] != 'N/D' and info['epoca'] != 'N/D':
            break

    # 3. Google Books: obras si aún hay menos de 3, y nacionalidad si aún falta
    if len(info['obras']) < 3 or info['nacionalidad'] == 'N/D':
        try:
            r_gb = requests.get(
                'https://www.googleapis.com/books/v1/volumes',
                params={'q': f'inauthor:"{nombre_autor}"',
                        'maxResults': 10,
                        'printType': 'books',
                        'orderBy': 'relevance'},
                timeout=10
            )
            titulos_vistos_gb = {t.lower() for t in info['obras']}
            for item in r_gb.json().get('items', []):
                vol_info = item.get('volumeInfo', {})
                titulo = vol_info.get('title', '').strip()
                if titulo and titulo.lower() not in titulos_vistos_gb:
                    titulos_vistos_gb.add(titulo.lower())
                    info['obras'].append(titulo)

                
                if info['nacionalidad'] == 'N/D':
                    descripcion = vol_info.get('description', '')
                    if descripcion:
                        for gentilicio, pais in _GENTILICIOS.items():
                            if re.search(rf'\b{re.escape(gentilicio)}\b', descripcion, re.IGNORECASE):
                                info['nacionalidad'] = pais
                                break

                if len(info['obras']) >= 6 and info['nacionalidad'] != 'N/D':
                    break
        except Exception:
            pass

    # 4. Gutendex: obras como último recurso si aún hay menos de 3
    if len(info['obras']) < 3:
        try:
            r_gut = requests.get(
                'https://gutendex.com/books/',
                params={'search': nombre_autor, 'languages': 'en,es'},
                timeout=10
            )
            titulos_vistos_gut = {t.lower() for t in info['obras']}
            apellido_buscado = _normalizar_nombre(nombre_autor.split()[-1])
            for libro in r_gut.json().get('results', [])[:10]:
                autores = libro.get('authors', [])
                if not autores:
                    continue
                nombre_libro = _invertir_nombre_gutenberg(autores[0].get('name', ''))
                if apellido_buscado not in _normalizar_nombre(nombre_libro):
                    continue
                titulo = libro.get('title', '').strip()
                if titulo and titulo.lower() not in titulos_vistos_gut:
                    titulos_vistos_gut.add(titulo.lower())
                    info['obras'].append(titulo)
                if len(info['obras']) >= 6:
                    break
        except Exception:
            pass

    return info


def _predecir_autor(texto: str) -> dict:
    autor_directo = _extraer_autor_del_texto(texto)
    if autor_directo:
        enriquecido = _enriquecer_autor(autor_directo)
        return {
            'autor':        autor_directo,
            'confianza':    'alta (texto)',
            'nacionalidad': enriquecido['nacionalidad'],
            'epoca':        enriquecido['epoca'],
            'obras':        enriquecido['obras'],
        }
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
        return {
            'autor':        'No identificado',
            'confianza':    'N/D',
            'nacionalidad': 'N/D',
            'epoca':        'N/D',
            'obras':        [],
        }

    autor, nivel = _consenso_autor(candidatos)
    enriquecido  = _enriquecer_autor(autor)
    return {
        'autor':        autor,
        'confianza':    nivel,
        'nacionalidad': enriquecido['nacionalidad'],
        'epoca':        enriquecido['epoca'],
        'obras':        enriquecido['obras'],
    }


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
            'nacionalidad':  ra['nacionalidad'],
            'epoca':         ra['epoca'],
            'obras':         ra['obras'],
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
        f"✍️ Autor: {resultado['autor']} ({resultado['conf_autor']})",
        f"   Nacionalidad: {resultado['nacionalidad']}",
        f"   Época: {resultado['epoca']}",
        f"   Obras representativas:",
    ]
    for i, obra in enumerate(resultado.get('obras', []), 1):
        lineas.append(f"     {i}. {obra}")
    if not resultado.get('obras'):
        lineas.append("     N/D")
    return "\n".join(lineas)