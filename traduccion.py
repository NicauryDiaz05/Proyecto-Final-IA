import os
from typing import Tuple, Dict, Any, List
from dotenv import load_dotenv
import evaluate
import deepl
import spacy
import nltk
from nltk.corpus import stopwords
from collections import Counter
load_dotenv()


# CONFIGURACIÓN DE APIs — Nicaury Díaz 23-SISN-2-028
DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY", "")
deepl_client  = deepl.Translator(DEEPL_API_KEY) if DEEPL_API_KEY else None


# MÉTRICAS — Nicaury Díaz 23-SISN-2-028
bertscore = evaluate.load("bertscore")
bleu      = evaluate.load("bleu")
rouge     = evaluate.load("rouge")


# MODELOS NLP — Nicaury Díaz 23-SISN-2-028
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

try:
    nlp_es = spacy.load("es_core_news_sm")
except OSError:
    os.system("python -m spacy download es_core_news_sm")
    nlp_es = spacy.load("es_core_news_sm")

try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp_en = spacy.load("en_core_web_sm")


# IDIOMAS FUENTE SOPORTADOS POR DEEPL 


LANG_MAP_SOURCE: Dict[str, str] = {
    "AR": "AR",    
    "BG": "BG",    
    "CS": "CS",    
    "DA": "DA",   
    "DE": "DE",   
    "EL": "EL",   
    "EN": "EN",    
    "ET": "ET",    
    "FI": "FI",   
    "FR": "FR",   
    "HE": "HE",   
    "HU": "HU",   
    "ID": "ID",    
    "IT": "IT",    
    "JA": "JA",    
    "KO": "KO",    
    "LT": "LT",   
    "LV": "LV",    
    "NB": "NB",   
    "NL": "NL",    
    "PL": "PL",    
    "PT": "PT",   
    "RO": "RO",    
    "RU": "RU",    
    "SK": "SK",   
    "SL": "SL",    
    "SV": "SV",    
    "TH": "TH",    
    "TR": "TR",    
    "UK": "UK",    
    "VI": "VI",    
    "ZH": "ZH",    
    "ZH-HANT": "ZH", 
}

# NOMBRES DE LOS IDIOMAS 
LANG_NAMES_ES: Dict[str, str] = {
    "AR":      "Árabe",
    "BG":      "Búlgaro",
    "CS":      "Checo",
    "DA":      "Danés",
    "DE":      "Alemán",
    "EL":      "Griego",
    "EN":      "Inglés",
    "ET":      "Estonio",
    "FI":      "Finlandés",
    "FR":      "Francés",
    "HE":      "Hebreo",
    "HU":      "Húngaro",
    "ID":      "Indonesio",
    "IT":      "Italiano",
    "JA":      "Japonés",
    "KO":      "Coreano",
    "LT":      "Lituano",
    "LV":      "Letón",
    "NB":      "Noruego (Bokmål)",
    "NL":      "Neerlandés",
    "PL":      "Polaco",
    "PT":      "Portugués",
    "RO":      "Rumano",
    "RU":      "Ruso",
    "SK":      "Eslovaco",
    "SL":      "Esloveno",
    "SV":      "Sueco",
    "TH":      "Tailandés",
    "TR":      "Turco",
    "UK":      "Ucraniano",
    "VI":      "Vietnamita",
    "ZH":      "Chino (simplificado)",
    "ZH-HANT": "Chino (tradicional)",
}

SOURCE_LANG_CHOICES: List[Tuple[str, str]] = sorted(
    [(nombre, codigo) for codigo, nombre in LANG_NAMES_ES.items()],
    key=lambda x: x[0]
)


TARGET_LANG_FIXED = "ES"



# TRADUCCIÓN BASE — DeepL Nicaury Díaz 23-SISN-2-028

def deepl_translate(text: str, source_lang: str) -> Tuple[str, str]:
    if not deepl_client:
        return "DeepL no disponible (falta API key).", "No se pudo llamar a DeepL."

    source_lang  = source_lang.upper().strip()
    deepl_source = LANG_MAP_SOURCE.get(source_lang, source_lang)

    try:
        result = deepl_client.translate_text(
            text,
            source_lang=deepl_source,
            target_lang=TARGET_LANG_FIXED,   
        )
        lang_nombre = LANG_NAMES_ES.get(source_lang, source_lang)
        return result.text, f"Traducido del {lang_nombre} al Español con DeepL."
    except Exception as e:
        return f"Error en DeepL: {str(e)}", "Falló la traducción base."


# ANÁLISIS ESTILÍSTICO —  Nicaury Díaz 23-SISN-2-028


def analizar_estilo(text: str, lang: str = "EN") -> Dict[str, Any]:
    try:
        nlp = nlp_es if lang.upper() == "ES" else nlp_en
        doc = nlp(text[:10000])

        oraciones   = list(doc.sents)
        n_oraciones = len(oraciones)
        long_prom   = round(sum(len(s) for s in oraciones) / max(n_oraciones, 1), 1)

        tokens   = [t.text.lower() for t in doc if t.is_alpha]
        n_tokens = len(tokens)
        n_unicos = len(set(tokens))
        riqueza  = round(n_unicos / max(n_tokens, 1) * 100, 1)

        sw_lang  = "spanish" if lang.upper() == "ES" else "english"
        sw       = set(stopwords.words(sw_lang))
        palabras = [t.lower() for t in tokens if t not in sw and len(t) > 2]
        top5     = Counter(palabras).most_common(5)

        entidades = {}
        for ent in doc.ents:
            entidades.setdefault(ent.label_, [])
            if ent.text not in entidades[ent.label_]:
                entidades[ent.label_].append(ent.text)

        pos_count = Counter(t.pos_ for t in doc if t.is_alpha)
        n_verbos  = pos_count.get("VERB", 0)
        n_adj     = pos_count.get("ADJ",  0)
        n_sust    = pos_count.get("NOUN", 0)

        adjetivos = [t.lemma_.lower() for t in doc if t.pos_ == "ADJ"]
        top_adj   = Counter(adjetivos).most_common(5)

        return {
            "n_palabras":   n_tokens,
            "n_oraciones":  n_oraciones,
            "long_prom":    long_prom,
            "riqueza":      riqueza,
            "n_unicos":     n_unicos,
            "top_palabras": top5,
            "top_adj":      top_adj,
            "entidades":    entidades,
            "verbos":       n_verbos,
            "adjetivos":    n_adj,
            "sustantivos":  n_sust,
            "error":        None,
        }
    except Exception as e:
        return {"error": str(e)}



# EVALUACIÓN DE CALIDAD —  Nicaury Díaz 23-SISN-2-028


def evaluate_translation(reference: str, candidate: str) -> Dict:
    bert   = bertscore.compute(predictions=[candidate], references=[reference], lang="es")
    bleu_  = bleu.compute(predictions=[candidate],  references=[[reference]])
    rouge_ = rouge.compute(predictions=[candidate], references=[reference])
    return {
        "bertscore_f1":   bert["f1"][0],
        "bleu":           bleu_["bleu"],
        "rouge_l":        rouge_["rougeL"],
        "quality_report": (
            f"BERTScore F1: {bert['f1'][0]:.3f}. "
            f"BLEU: {bleu_['bleu']:.3f}. "
            f"ROUGE-L: {rouge_['rougeL']:.3f}."
        )
    }


# PIPELINE COMPLETO — Nicaury Díaz 23-SISN-2-028

def full_translation_pipeline(
    text: str,
    source_lang: str,
) -> Dict[str, Any]:
    base_trans, deepl_notes = deepl_translate(text, source_lang)
    estilo    = analizar_estilo(text, lang=source_lang)
    eval_data = evaluate_translation(text, base_trans)

    return {
        "deepl":  {"translation": base_trans, "notes": deepl_notes},
        "estilo": estilo,
        "eval":   eval_data,
    }



# FORMATO HTML PARA GRADIO — Nicaury Díaz 23-SISN-2-028


def format_deepl(d: dict) -> str:
    return (
        f"<p><strong>Traducción al Español (DeepL):</strong><br>{d['translation']}</p>"
        f"<p><em>{d['notes']}</em></p>"
    )


def format_estilo(g: dict) -> str:
    if g.get("error"):
        return f"<p>❌ Error en análisis: {g['error']}</p>"

    def fila(label, valor):
        return (
            f"<tr>"
            f"<td style='padding:4px 12px 4px 0;color:#FCE7F3;'><strong>{label}</strong></td>"
            f"<td style='color:#FCE7F3;'>{valor}</td>"
            f"</tr>"
        )

    top_pal = ", ".join(f"{w} ({c})" for w, c in g.get("top_palabras", []))
    top_adj = ", ".join(f"{w} ({c})" for w, c in g.get("top_adj", []))

    entidades_html = ""
    for tipo, lista in g.get("entidades", {}).items():
        entidades_html += f"<li><strong>{tipo}:</strong> {', '.join(lista[:4])}</li>"

    html  = "<h3 style='color:#EC4899;'>✨ Análisis Estilístico</h3>"
    html += "<table style='width:100%;border-collapse:collapse;'>"
    html += fila("📝 Palabras",       g.get("n_palabras",  "N/D"))
    html += fila("📖 Oraciones",      g.get("n_oraciones", "N/D"))
    html += fila("📏 Long. promedio", f"{g.get('long_prom', 'N/D')} chars/oración")
    html += fila("💎 Riqueza léxica", f"{g.get('riqueza', 'N/D')}% ({g.get('n_unicos','N/D')} palabras únicas)")
    html += fila("🔤 Sustantivos",    g.get("sustantivos", "N/D"))
    html += fila("⚡ Verbos",         g.get("verbos",      "N/D"))
    html += fila("🎨 Adjetivos",      g.get("adjetivos",   "N/D"))
    html += fila("🔑 Palabras clave", top_pal or "—")
    html += fila("🌈 Top adjetivos",  top_adj or "—")
    html += "</table>"

    if entidades_html:
        html += (
            f"<h4 style='color:#EC4899;margin-top:12px;'>🗺️ Entidades detectadas</h4>"
            f"<ul style='color:#FCE7F3;'>{entidades_html}</ul>"
        )
    return html


def format_eval(e: dict) -> str:
    html  = "<h3 style='color:#EC4899;'>📈 Métricas de calidad</h3>"
    html += "<ul style='color:#FCE7F3;'>"
    html += f"<li>BERTScore F1: {e['bertscore_f1']:.3f}</li>"
    html += f"<li>BLEU: {e['bleu']:.3f}</li>"
    html += f"<li>ROUGE-L: {e['rouge_l']:.3f}</li>"
    html += f"</ul><p style='color:#FCE7F3;'>{e['quality_report']}</p>"
    return html