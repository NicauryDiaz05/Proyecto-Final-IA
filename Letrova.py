import re
import gradio as gr
import constante
import ocr
import clasificador
import traduccion
import docx
import tempfile
import database 
import pdfplumber   
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from traduccion import SOURCE_LANG_CHOICES


# CSS personalizado Nicaury Diaz 23-SISN-2-028
css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@1,700&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #0A0A0F !important;
    font-family: 'DM Sans', sans-serif !important;
}

.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 0%, #1E1E4A 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, #12122A 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 50% 50%, #0f0f25 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

footer { display: none !important; }

#titulo { text-align: center; padding: 2rem 0 1rem; }
#titulo h1 {
    font-family: 'Playfair Display', serif !important;
    font-style: italic !important;
    font-size: 3.2rem !important;
    background: linear-gradient(135deg, #FCE7F3, #EC4899, #FCE7F3) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 0 0.3rem !important;
    animation: glow 3s ease-in-out infinite alternate;
}
@keyframes glow {
    from { filter: drop-shadow(0 0 8px rgba(236,72,153,0.4)); }
    to   { filter: drop-shadow(0 0 22px rgba(252,231,243,0.9)); }
}
#titulo p { color: #FCE7F3 !important; font-size: 0.95rem !important; letter-spacing: 0.08em; margin: 0 !important; }

.tab-nav {
    background: rgba(18,18,42,0.6) !important;
    border: 1px solid rgba(236,72,153,0.2) !important;
    border-radius: 50px !important;
    padding: 5px !important;
    backdrop-filter: blur(20px) !important;
}
.tab-nav button {
    border-radius: 50px !important;
    color: #FCE7F3 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.3s !important;
    border: none !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #1E1E4A, #EC4899) !important;
    color: #FCE7F3 !important;
    box-shadow: 0 4px 15px rgba(236,72,153,0.5) !important;
}

.panel {
    background: rgba(18,18,42,0.4) !important;
    border: 1px solid rgba(236,72,153,0.15) !important;
    border-radius: 24px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(20px) !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
.panel:hover {
    border-color: rgba(236,72,153,0.5) !important;
    box-shadow: 0 0 35px rgba(236,72,153,0.15) !important;
}

textarea, input {
    background: rgba(18,18,42,0.5) !important;
    border: 1px solid rgba(236,72,153,0.2) !important;
    border-radius: 14px !important;
    color: #FCE7F3 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
textarea:focus, input:focus {
    border-color: rgba(236,72,153,0.7) !important;
    box-shadow: 0 0 0 3px rgba(236,72,153,0.15) !important;
}
textarea::placeholder { color: rgba(252,231,243,0.35) !important; }
label span { color: #FCE7F3 !important; font-weight: 500 !important; font-size: 0.9rem !important; }

button.primary {
    background: linear-gradient(135deg, #1E1E4A, #EC4899) !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 12px 32px !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 20px rgba(236,72,153,0.4) !important;
    transition: all 0.3s !important;
}
button.primary:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 30px rgba(236,72,153,0.6) !important;
}
button.primary:active { transform: scale(0.98) !important; }

.upload-container {
    border: 2px dashed rgba(236,72,153,0.25) !important;
    border-radius: 18px !important;
    background: rgba(18,18,42,0.2) !important;
}

/* Historial */
.historial-tabla {
    width: 100%;
    border-collapse: collapse;
    color: #FCE7F3;
    font-size: 0.85rem;
}
.historial-tabla th {
    background: rgba(236,72,153,0.25);
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.historial-tabla td {
    padding: 7px 12px;
    border-bottom: 1px solid rgba(236,72,153,0.1);
    vertical-align: top;
}
.historial-tabla tr:hover td { background: rgba(236,72,153,0.07); }
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-texto   { background: rgba( 59,130,246,0.25); color:#93C5FD; }
.badge-archivo { background: rgba(234,179,  8,0.25); color:#FDE68A; }
.badge-imagen  { background: rgba(167, 80,162,0.25); color:#E9D5FF; }

#footer { text-align: center; padding: 1rem 0 2rem; }
#footer p { color: rgba(236,72,153,0.35) !important; font-size: 0.8rem !important; letter-spacing: 0.1em; }
"""


# Patrón para detectar encabezados de capítulo
PATRON_CAPITULO = re.compile(
    r'(?:^|\n)((?:cap[íi]tulo|chapter|parte|part|secci[oó]n|section'
    r'|libro|book|pr[oó]logo|prologue|ep[íi]logo|epilogue)'
    r'[\s\w\.,:\-–—]*)',
    flags=re.IGNORECASE
)

PATRON_ES_CAPITULO = re.compile(
    r'^(cap[íi]tulo|chapter|parte|part|secci[oó]n|section'
    r'|libro|book|pr[oó]logo|prologue|ep[íi]logo|epilogue)',
    re.IGNORECASE
)


# extraer la clasificación ─
def _parsear_clasificacion(clasif_texto: str) -> tuple[str, str, str, float]:
    genero     = "Sin clasificar"
    tipo       = "Sin clasificar"
    autor      = "Desconocido"
    confianza  = 0.0

    for linea in clasif_texto.splitlines():
        linea_lower = linea.lower()
        if "género" in linea_lower or "genero" in linea_lower:
            partes = linea.split(":", 1)
            if len(partes) == 2:
                genero = partes[1].strip()
        elif "tipo" in linea_lower or "lectura" in linea_lower:
            partes = linea.split(":", 1)
            if len(partes) == 2:
                tipo = partes[1].strip()
        elif "autor" in linea_lower:
            partes = linea.split(":", 1)
            if len(partes) == 2:
                autor = partes[1].strip()
        elif "confianza" in linea_lower or "confidence" in linea_lower or "%" in linea:
            # busca el primer número (entero o decimal)
            nums = re.findall(r'\d+\.?\d*', linea)
            if nums:
                val = float(nums[0])
                confianza = val / 100.0 if val > 1 else val

    return genero, tipo, autor, confianza


#Reconstrucción de párrafos
def _reconstruir_parrafos(lineas):
    resultado = []
    buffer = ""

    for linea in lineas:
        linea = linea.strip()

        if not linea:
            if buffer:
                resultado.append(buffer)
                buffer = ""
            resultado.append("")
            continue

        if PATRON_ES_CAPITULO.match(linea):
            if buffer:
                resultado.append(buffer)
                buffer = ""
            resultado.append(linea)
            continue

        if buffer:
            termina_frase = buffer[-1] in '.!?»"\u201d'
            if termina_frase and linea[0].isupper():
                resultado.append(buffer)
                buffer = linea
            else:
                buffer = buffer + " " + linea
        else:
            buffer = linea

    if buffer:
        resultado.append(buffer)
    return resultado


def _colapsar_blancos(parrafos):
    final = []
    prev_blank = False
    for p in parrafos:
        if p == "":
            if not prev_blank:
                final.append("")
            prev_blank = True
        else:
            final.append(p)
            prev_blank = False
    return final


def limpiar_texto_pdf(texto_crudo):
    lineas = texto_crudo.splitlines()
    return "\n".join(_colapsar_blancos(_reconstruir_parrafos(lineas)))


def limpiar_texto_traducido(texto):
    lineas = [l.strip() for l in texto.splitlines()]
    return "\n".join(_colapsar_blancos(_reconstruir_parrafos(lineas)))


# Segmentación por capítulos 
def _subdividir_cuerpo(encabezado, cuerpo, max_palabras):
    parrafos = [p.strip() for p in cuerpo.split('\n\n') if p.strip()]
    resultado = []
    buffer = []
    buffer_palabras = 0
    primer_chunk = True

    for p in parrafos:
        palabras_p = len(p.split())
        if buffer_palabras + palabras_p > max_palabras and buffer:
            enc = encabezado if primer_chunk else f"{encabezado} (cont.)"
            resultado.append((enc, "\n\n".join(buffer)))
            buffer = []
            buffer_palabras = 0
            primer_chunk = False
        buffer.append(p)
        buffer_palabras += palabras_p

    if buffer:
        enc = encabezado if primer_chunk else f"{encabezado} (cont.)"
        resultado.append((enc, "\n\n".join(buffer)))

    return resultado


def segmentar_por_capitulos(texto, max_palabras=800):
    posiciones = [(m.start(), m.group(1).strip()) for m in PATRON_CAPITULO.finditer(texto)]

    if not posiciones:
        parrafos = [p.strip() for p in texto.split('\n\n') if p.strip()]
        resultado = []
        buffer = []
        buffer_palabras = 0

        for p in parrafos:
            palabras_p = len(p.split())
            if buffer_palabras + palabras_p > max_palabras and buffer:
                resultado.append(("", "\n\n".join(buffer)))
                buffer = []
                buffer_palabras = 0
            buffer.append(p)
            buffer_palabras += palabras_p

        if buffer:
            resultado.append(("", "\n\n".join(buffer)))

        return resultado

    segmentos = []
    for i, (pos, encabezado) in enumerate(posiciones):
        inicio_cuerpo = pos + len(encabezado)
        fin_cuerpo = posiciones[i + 1][0] if i + 1 < len(posiciones) else len(texto)
        cuerpo = texto[inicio_cuerpo:fin_cuerpo].strip()

        if len(cuerpo.split()) <= max_palabras:
            segmentos.append((encabezado, cuerpo))
        else:
            segmentos.extend(_subdividir_cuerpo(encabezado, cuerpo, max_palabras))

    return segmentos


# Generación de PDF 
def generar_pdf_descarga(texto_traducido, titulo="Traduccion"):
    if not texto_traducido or not texto_traducido.strip():
        return None
    texto_traducido = limpiar_texto_traducido(texto_traducido)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()

    PAGE_W, PAGE_H = A4
    MARGEN_H = 3.2 * cm
    MARGEN_V = 2.8 * cm
    COLOR_TITULO  = colors.HexColor("#1a1a2e")
    COLOR_CUERPO  = colors.HexColor("#1c1c1c")
    COLOR_LINEA   = colors.HexColor("#c084fc")
    COLOR_HEADER  = colors.HexColor("#6b21a8")

    titulo_corto = titulo[:60] + "..." if len(titulo) > 60 else titulo

    def _header_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(COLOR_LINEA)
        canvas.setLineWidth(0.8)
        canvas.line(MARGEN_H, PAGE_H - MARGEN_V + 0.4*cm,
                    PAGE_W - MARGEN_H, PAGE_H - MARGEN_V + 0.4*cm)
        canvas.setFont("Helvetica-Oblique", 8)
        canvas.setFillColor(COLOR_HEADER)
        canvas.drawString(MARGEN_H, PAGE_H - MARGEN_V + 0.6*cm, titulo_corto)
        canvas.line(MARGEN_H, MARGEN_V - 0.4*cm,
                    PAGE_W - MARGEN_H, MARGEN_V - 0.4*cm)
        # Número de página centrado
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#9ca3af"))
        canvas.drawCentredString(PAGE_W / 2, MARGEN_V - 0.7*cm,
                                 str(doc.page))
        canvas.restoreState()

    frame = Frame(
        MARGEN_H, MARGEN_V,
        PAGE_W - 2 * MARGEN_H,
        PAGE_H - 2 * MARGEN_V,
        leftPadding=0, rightPadding=0,
        topPadding=0.3*cm, bottomPadding=0.3*cm,
    )
    template = PageTemplate(id="letrova", frames=[frame],
                            onPage=_header_footer)
    doc = BaseDocTemplate(
        tmp.name,
        pagesize=A4,
        leftMargin=MARGEN_H, rightMargin=MARGEN_H,
        topMargin=MARGEN_V,  bottomMargin=MARGEN_V,
    )
    doc.addPageTemplates([template])

    estilo_titulo_doc = ParagraphStyle(
        "TituloDoc",
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=COLOR_TITULO,
        spaceAfter=6,
        leading=26,
    )
    estilo_subtitulo = ParagraphStyle(
        "Subtitulo",
        fontName="Helvetica-Oblique",
        fontSize=10,
        textColor=colors.HexColor("#7c3aed"),
        spaceAfter=20,
    )
    estilo_capitulo = ParagraphStyle(
        "Capitulo",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=COLOR_TITULO,
        spaceBefore=22,
        spaceAfter=8,
        leading=18,
    )
    estilo_cuerpo = ParagraphStyle(
        "Cuerpo",
        fontName="Helvetica",
        fontSize=11,
        leading=19,
        textColor=COLOR_CUERPO,
        spaceAfter=8,
        firstLineIndent=18,
    )

    story = []

    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph(titulo, estilo_titulo_doc))
    story.append(Paragraph("Traducción generada por Letrova", estilo_subtitulo))
    story.append(HRFlowable(width="100%", thickness=0.8,
                             color=COLOR_LINEA, spaceAfter=18))

    for parrafo in texto_traducido.split("\n"):
        parrafo = parrafo.strip()
        if not parrafo:
            story.append(Spacer(1, 6))
            continue

        parrafo_limpio = (
            parrafo
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        if PATRON_ES_CAPITULO.match(parrafo):
            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="30%", thickness=0.5,
                                     color=COLOR_LINEA, spaceAfter=4))
            story.append(Paragraph(parrafo_limpio, estilo_capitulo))
        else:
            story.append(Paragraph(parrafo_limpio, estilo_cuerpo))

    doc.build(story)
    return tmp.name


# Historial UI 
def _fila_html(row: dict) -> str:
    badge_class = {
        "texto":   "badge-texto",
        "archivo": "badge-archivo",
        "imagen":  "badge-imagen",
    }.get(row.get("fuente", "texto"), "badge-texto")

    icono = {"texto": "✏️", "archivo": "📁", "imagen": "📷"}.get(row.get("fuente"), "✏️")

    conf_pct = f"{row['confianza']*100:.0f}%" if row['confianza'] else "—"

    return (
        f"<tr>"
        f"<td><span class='badge {badge_class}'>{icono} {row.get('fuente','')}</span></td>"
        f"<td>{row.get('genero','')}</td>"
        f"<td>{row.get('tipo','')}</td>"
        f"<td>{row.get('autor','')}</td>"
        f"<td>{conf_pct}</td>"
        f"<td style='color:rgba(252,231,243,0.5);font-size:0.78rem'>{row.get('fecha','')[:19]}</td>"
        f"<td style='max-width:220px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis'>{row.get('texto','')}</td>"
        f"</tr>"
    )


def cargar_historial():
    datos = database.obtener_historial(limit=30)
    if not datos:
        return "<p style='color:rgba(252,231,243,0.4);text-align:center;padding:2rem'>No hay análisis guardados aún.</p>"

    filas = "".join(_fila_html(r) for r in datos)
    return (
        "<table class='historial-tabla'>"
        "<thead><tr>"
        "<th>Fuente</th><th>Género</th><th>Tipo</th><th>Autor</th>"
        "<th>Confianza</th><th>Fecha</th><th>Fragmento</th>"
        "</tr></thead>"
        f"<tbody>{filas}</tbody>"
        "</table>"
    )


def limpiar_historial():
    msg = database.borrar_historial()
    return cargar_historial(), msg


#  Funciones principales 
def analizar_y_traducir(texto, src):
    if not texto or not texto.strip():
        return constante.ERROR_TEXTO, "", "<p></p>", "<p></p>", "<p></p>", None

    palabras = len(texto.split())
    resumen = (
        f"✨ Texto recibido — {len(texto)} caracteres · "
        f"{palabras} palabras · listo para procesar."
    )

    resultado = clasificador.clasificar_texto(texto)
    clasif = clasificador.resultado_texto(resultado)

    res = traduccion.full_translation_pipeline(texto, src)  

    texto_traducido = res["deepl"].get("translation", "")
    pdf = generar_pdf_descarga(texto_traducido, titulo="Traduccion - Letrova")

    genero, tipo, autor, confianza = _parsear_clasificacion(clasif)
    database.guardar_resultado(
        texto=texto,
        genero=genero,
        tipo=tipo,
        autor=autor,
        confianza=confianza,
        fuente="texto",
    )

    return (
        resumen, clasif,
        traduccion.format_deepl(res["deepl"]),
        traduccion.format_estilo(res["estilo"]),  
        traduccion.format_eval(res["eval"]),
        pdf
    )


def procesar_archivo(archivo, src):
    if archivo is None:
        return constante.ERROR_ARCHIVO, "", "<p></p>", "<p></p>", "<p></p>", None

    nombre = archivo.name.split("\\")[-1].split("/")[-1]
    extension = "." + nombre.split(".")[-1].lower()

    if extension not in constante.TIPOS_VALIDOS:
        return constante.ERROR_FORMATO, "", "<p></p>", "<p></p>", "<p></p>", None

    try:
        texto = ""

        if extension == ".txt":
            with open(archivo.name, "r", encoding="utf-8", errors="ignore") as f:
                texto = f.read()

        elif extension == ".pdf":
            texto_crudo = ""
            with pdfplumber.open(archivo.name) as pdf:
                for page in pdf.pages:
                    contenido = page.extract_text()
                    if contenido:
                        texto_crudo += contenido + "\n"
            texto = limpiar_texto_pdf(texto_crudo)

        elif extension == ".docx":
            doc = docx.Document(archivo.name)
            texto = "\n".join([p.text for p in doc.paragraphs])

        elif extension == ".epub":
            from ebooklib import epub, ITEM_DOCUMENT
            from bs4 import BeautifulSoup
            book = epub.read_epub(archivo.name)
            for item in book.get_items():
                if item.get_type() == ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    texto += soup.get_text()

        segmentos = segmentar_por_capitulos(texto, max_palabras=800)
        total_segmentos = len(segmentos)

        resultado_global = clasificador.clasificar_texto(texto[:5000])
        clasificacion_final = clasificador.resultado_texto(resultado_global)

        estilo_global = traduccion.analizar_estilo(texto[:5000], lang=src)
        gpt_final = traduccion.format_estilo(estilo_global)  

        partes_traducidas = []
        for encabezado, cuerpo in segmentos[:10]:
            if not cuerpo.strip():
                continue

            res = traduccion.full_translation_pipeline(cuerpo, src)  
            traduccion_cuerpo = res["deepl"].get("translation", "")

            if encabezado:
                partes_traducidas.append(f"{encabezado}\n\n{traduccion_cuerpo}")
            else:
                partes_traducidas.append(traduccion_cuerpo)

        texto_completo_traducido = "\n\n\n".join(partes_traducidas)

        muestra_original = " ".join(texto.split()[:300])
        muestra_traduccion = " ".join(texto_completo_traducido.split()[:300])
        eval_global = traduccion.evaluate_translation(muestra_original, muestra_traduccion)
        eval_final = traduccion.format_eval(eval_global)

        deepl_dict = {
            "translation": texto_completo_traducido,
            "notes": f"Traducción DeepL completa · {total_segmentos} segmentos procesados."
        }
        deepl_final = traduccion.format_deepl(deepl_dict)

        nombre_base = nombre.rsplit(".", 1)[0]
        pdf_descarga = generar_pdf_descarga(
            texto_completo_traducido,
            titulo=f"Traduccion de: {nombre_base}"
        )

        genero, tipo, autor, confianza = _parsear_clasificacion(clasificacion_final)
        database.guardar_resultado(
            texto=texto,
            genero=genero,
            tipo=tipo,
            autor=autor,
            confianza=confianza,
            fuente="archivo",
        )

        chunks_usados = min(total_segmentos, 10)
        return (
            f"📄 Archivo procesado: {nombre} · "
            f"{len(texto.split())} palabras · "
            f"{total_segmentos} segmentos detectados · "
            f"{chunks_usados} traducidos",
            clasificacion_final,
            deepl_final,
            gpt_final,
            eval_final,
            pdf_descarga
        )

    except Exception as e:
        return f"❌ Error al procesar el archivo: {str(e)}", "", "<p></p>", "<p></p>", "<p></p>", None


def procesar_imagen(imagen, src):
    if imagen is None:
        return constante.ERROR_IMAGEN, "", "<p></p>", "<p></p>", "<p></p>", None
    try:
        texto, clasificacion = ocr.extraer_texto(imagen)
        if not texto.strip():
            return (
                "⚠️ No se detectó texto en la imagen. Intenta con una foto más nítida.",
                "", "<p></p>", "<p></p>", "<p></p>", None
            )

        palabras = len(texto.split())
        resumen = f"📖 Texto extraído — {len(texto)} caracteres · {palabras} palabras\n\n{texto}"

        res = traduccion.full_translation_pipeline(texto, src)  

        texto_traducido = res["deepl"].get("translation", "")
        pdf = generar_pdf_descarga(texto_traducido, titulo="Traduccion de imagen - Letrova")

        genero, tipo, autor, confianza = _parsear_clasificacion(clasificacion)
        database.guardar_resultado(
            texto=texto,
            genero=genero,
            tipo=tipo,
            autor=autor,
            confianza=confianza,
            fuente="imagen",
        )

        return (
            resumen, clasificacion,
            traduccion.format_deepl(res["deepl"]),
            traduccion.format_estilo(res["estilo"]),  
            traduccion.format_eval(res["eval"]),
            pdf
        )
    except Exception as e:
        return f"❌ Error al procesar la imagen: {str(e)}", "", "<p></p>", "<p></p>", "<p></p>", None


# Interfaz 
with gr.Blocks(title=constante.APP_TITLE) as app:

    gr.HTML(f'<div id="titulo"><h1>{constante.TITULO}</h1><p>{constante.SUBTITULO}</p></div>')

    with gr.Tabs():

        # Pestaña Texto
        with gr.Tab("✏️ Texto"):
            with gr.Group(elem_classes="panel"):
                texto_input = gr.Textbox(
                    label="✨ Entrada de texto",
                    placeholder="Escribe o pega el texto...",
                    lines=7
                )
                with gr.Row():
                    t_src = gr.Dropdown(
                        label="Idioma origen",
                        choices=SOURCE_LANG_CHOICES,
                        value="EN"
                    )
                    t_tgt = gr.Textbox(
                        label="Idioma destino",
                        value="Español",
                        interactive=False
                    )
                texto_btn    = gr.Button("💗 Analizar y traducir", variant="primary")
                texto_salida = gr.Textbox(label="Resultado", interactive=False, lines=2)
                texto_clasif = gr.Textbox(label="🌸 Clasificación literaria", interactive=False, lines=14)
                with gr.Tabs():
                    with gr.Tab("🔤 DeepL"):      t_deepl = gr.HTML()
                    with gr.Tab("✨ Análisis"):    t_gpt   = gr.HTML()
                    with gr.Tab("📈 Evaluación"): t_eval  = gr.HTML()
                t_descarga = gr.File(label="⬇️ Descargar traducción en PDF", interactive=False)
            texto_btn.click(
                analizar_y_traducir,
                inputs=[texto_input, t_src],
                outputs=[texto_salida, texto_clasif, t_deepl, t_gpt, t_eval, t_descarga]
            )

        # Pestaña Archivos 
        with gr.Tab("📁 Archivos"):
            with gr.Group(elem_classes="panel"):
                archivo_input = gr.File(
                    label="📄 Subir archivo (PDF · EPUB · DOCX · TXT)",
                    file_types=[".pdf", ".epub", ".docx", ".txt"]
                )
                with gr.Row():
                    a_src = gr.Dropdown(
                        label="Idioma origen",
                        choices=SOURCE_LANG_CHOICES,
                        value="EN"
                    )
                    a_tgt = gr.Textbox(
                        label="Idioma destino",
                        value="Español",
                        interactive=False
                    )
                archivo_btn    = gr.Button("🌸 Analizar y traducir", variant="primary")
                archivo_salida = gr.Textbox(label="Resultado", interactive=False, lines=2)
                archivo_clasif = gr.Textbox(label="🌸 Clasificación literaria", interactive=False, lines=14)
                with gr.Tabs():
                    with gr.Tab("🔤 DeepL"):      a_deepl = gr.HTML()
                    with gr.Tab("✨ Análisis"):    a_gpt   = gr.HTML()
                    with gr.Tab("📈 Evaluación"): a_eval  = gr.HTML()
                a_descarga = gr.File(label="⬇️ Descargar traducción en PDF", interactive=False)
            archivo_btn.click(
                procesar_archivo,
                inputs=[archivo_input, a_src],
                outputs=[archivo_salida, archivo_clasif, a_deepl, a_gpt, a_eval, a_descarga]
            )

        # Pestaña Cámara 
        with gr.Tab("📷 Cámara"):
            with gr.Group(elem_classes="panel"):
                imagen_input = gr.Image(
                    label="📸 Captura desde cámara o sube una imagen",
                    type="pil",
                    sources=["upload", "webcam"]
                )
                with gr.Row():
                    c_src = gr.Dropdown(
                        label="Idioma origen",
                        choices=SOURCE_LANG_CHOICES,
                        value="EN"
                    )
                    c_tgt = gr.Textbox(
                        label="Idioma destino",
                        value="Español",
                        interactive=False
                    )
                imagen_btn    = gr.Button("💜 Analizar y traducir", variant="primary")
                imagen_salida = gr.Textbox(label="Resultado", interactive=False, lines=2)
                imagen_clasif = gr.Textbox(label="🌸 Clasificación literaria", interactive=False, lines=14)
                with gr.Tabs():
                    with gr.Tab("🔤 DeepL"):      c_deepl = gr.HTML()
                    with gr.Tab("✨ Análisis"):    c_gpt   = gr.HTML()
                    with gr.Tab("📈 Evaluación"): c_eval  = gr.HTML()
                c_descarga = gr.File(label="⬇️ Descargar traducción en PDF", interactive=False)
            imagen_btn.click(
                procesar_imagen,
                inputs=[imagen_input, c_src],
                outputs=[imagen_salida, imagen_clasif, c_deepl, c_gpt, c_eval, c_descarga]
            )

        # Pestaña Historial 
        with gr.Tab("📊 Historial"):
            with gr.Group(elem_classes="panel"):
                with gr.Row():
                    hist_btn_cargar = gr.Button("🔄 Actualizar historial", variant="primary")
                    hist_btn_borrar = gr.Button("🗑️ Borrar historial", variant="secondary")
                hist_msg    = gr.Textbox(label="", interactive=False, visible=True, lines=1)
                hist_tabla  = gr.HTML(value=cargar_historial())  

            hist_btn_cargar.click(
                fn=cargar_historial,
                inputs=[],
                outputs=[hist_tabla]
            )
            hist_btn_borrar.click(
                fn=limpiar_historial,
                inputs=[],
                outputs=[hist_tabla, hist_msg]
            )

    gr.HTML(f'<div id="footer"><p>{constante.FOOTER}</p></div>')

if __name__ == "__main__":
    app.launch(css=css)