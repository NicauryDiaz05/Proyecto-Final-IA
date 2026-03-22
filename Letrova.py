import gradio as gr
import constante
import ocr

#CSS personalizado Nicaury Diaz 23-SISN-2-028
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

#footer { text-align: center; padding: 1rem 0 2rem; }
#footer p { color: rgba(236,72,153,0.35) !important; font-size: 0.8rem !important; letter-spacing: 0.1em; }
"""
#Funciones Principales — validan y reciben los 3 canales de entrada Nicaury Diaz 23-SISN-2-028
def analizar_texto(texto):
    if not texto or not texto.strip():
        return constante.ERROR_TEXTO
    palabras = len(texto.split())
    return f"✨ Texto recibido — {len(texto)} caracteres · {palabras} palabras · listo para procesar."

def procesar_archivo(archivo):
    if archivo is None:
        return constante.ERROR_ARCHIVO
    nombre    = archivo.name.split("\\")[-1].split("/")[-1]
    extension = "." + nombre.split(".")[-1].lower()
    if extension not in constante.TIPOS_VALIDOS:
        return constante.ERROR_FORMATO
    return f"📄 Archivo recibido: {nombre} · listo para procesar."

def procesar_imagen(imagen):
    if imagen is None:
        return constante.ERROR_IMAGEN
    try:
        texto = ocr.extraer_texto(imagen)
        if not texto.strip():
            return "⚠️ No se detectó texto en la imagen. Intenta con una foto más nítida."
        palabras = len(texto.split())
        return f"📖 Texto extraído — {len(texto)} caracteres · {palabras} palabras\n\n{texto}"
    except Exception as e:
        return f"❌ Error al procesar la imagen: {str(e)}"

# Interfaz gr.blocks que permite que la aplicacion sea funcional Nicaury Diaz 23-SISN-2-028
with gr.Blocks(title=constante.APP_TITLE) as app:

    gr.HTML(f'<div id="titulo"><h1>{constante.TITULO}</h1><p>{constante.SUBTITULO}</p></div>')

    with gr.Tabs():
        with gr.Tab("✏️ Texto"):
            with gr.Group(elem_classes="panel"):
                texto_input  = gr.Textbox(label="✨ Entrada de texto", placeholder="Escribe o pega el texto...", lines=7)
                texto_btn    = gr.Button("💗 Analizar texto", variant="primary")
                texto_salida = gr.Textbox(label="Resultado", interactive=False, lines=2)
            texto_btn.click(analizar_texto, texto_input, texto_salida)

        with gr.Tab("📁 Archivos"):
            with gr.Group(elem_classes="panel"):
                archivo_input  = gr.File(label="📄 Subir archivo (PDF · EPUB · DOCX · TXT)", file_types=[".pdf",".epub",".docx",".txt"])
                archivo_btn    = gr.Button("🌸 Procesar archivo", variant="primary")
                archivo_salida = gr.Textbox(label="Resultado", interactive=False, lines=2)
            archivo_btn.click(procesar_archivo, archivo_input, archivo_salida)

        with gr.Tab("📷 Cámara"):
            with gr.Group(elem_classes="panel"):
                imagen_input  = gr.Image(label="📸 Captura desde cámara o sube una imagen", type="pil", sources=["upload","webcam"])
                imagen_btn    = gr.Button("💜 Procesar imagen", variant="primary")
                imagen_salida = gr.Textbox(label="Resultado", interactive=False, lines=2)
            imagen_btn.click(procesar_imagen, imagen_input, imagen_salida)

    gr.HTML(f'<div id="footer"><p>{constante.FOOTER}</p></div>')

if __name__ == "__main__":
    app.launch(css=css)