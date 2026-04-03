import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL no está definido en el archivo .env")

engine = create_engine(DATABASE_URL)

# CREAR TABLA DE HISTORIAL

def crear_tabla():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS analisis_libros (
                id            SERIAL PRIMARY KEY,
                texto         TEXT,
                genero        VARCHAR(100),
                tipo_lectura  VARCHAR(100),
                autor         VARCHAR(200),
                confianza     FLOAT,
                fuente        VARCHAR(20)  DEFAULT 'texto',
                fecha         TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.commit()

try:
    crear_tabla()
except Exception as _e:
    print(f"[database] No se pudo crear la tabla: {_e}")



# GUARDAR RESULTADO

def guardar_resultado(
    texto: str,
    genero: str,
    tipo: str,
    autor: str = "Desconocido",
    confianza: float = 0.0,
    fuente: str = "texto",          # "texto" | "archivo" | "imagen"
):
    
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO analisis_libros
                    (texto, genero, tipo_lectura, autor, confianza, fuente)
                VALUES
                    (:texto, :genero, :tipo, :autor, :confianza, :fuente)
            """), {
                "texto":     (texto or "")[:2000],
                "genero":    genero   or "Sin clasificar",
                "tipo":      tipo     or "Sin clasificar",
                "autor":     autor    or "Desconocido",
                "confianza": confianza or 0.0,
                "fuente":    fuente,
            })
            conn.commit()
    except Exception as e:
        print(f"[database] Error al guardar resultado: {e}")



# OBTENER HISTORIAL

def obtener_historial(limit: int = 20) -> list[dict]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT id, texto, genero, tipo_lectura, autor,
                       confianza, fuente, fecha
                FROM   analisis_libros
                ORDER  BY fecha DESC
                LIMIT  :limit
            """), {"limit": limit}).mappings()

            return [
                {
                    "id":       row["id"],
                    "texto":    (row["texto"][:120] + "…") if row["texto"] else "",
                    "genero":   row["genero"],
                    "tipo":     row["tipo_lectura"],
                    "autor":    row["autor"],
                    "confianza": row["confianza"],
                    "fuente":   row["fuente"],
                    "fecha":    str(row["fecha"]),
                }
                for row in rows
            ]
    except Exception as e:
        print(f"[database] Error al obtener historial: {e}")
        return []



# BORRAR HISTORIAL

def borrar_historial():
    try:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM analisis_libros;"))
            conn.commit()
        return "🗑️ Historial borrado correctamente."
    except Exception as e:
        print(f"[database] Error al borrar historial: {e}")
        return f"❌ Error al borrar: {e}"