import os
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


def _predecir_autor(texto: str) -> dict:
    if _embedder is None or _autores_index is None:
        return {'autor': 'No identificado', 'confianza': 'N/D'}
    emb = _embedder.encode([texto], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(emb)
    D, I = _autores_index.search(emb, k=1)
    similitud = float(D[0][0])
    autor = _autores_meta[I[0][0]] if similitud > 0.5 else 'No identificado'
    return {'autor': autor, 'confianza': f'{similitud*100:.1f}%' if similitud > 0.5 else 'baja'}



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