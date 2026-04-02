# ============================================================
# CABECERA
# ============================================================
# Alumno: Mariona Macià Cuartero
# URL Streamlit Cloud: https://mda13-bc5-spotify-gh2tpq92tgdfccappkflqhy.streamlit.app/
# URL GitHub: https://github.com/nonamacia-source/mda13-bc5-spotify.git

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un asistente especializado en análisis de datos de Spotify mediante generación de código Python.

Recibes preguntas en lenguaje natural sobre el historial de escucha de un usuario.

Tu tarea NO es responder directamente, sino generar código Python que analice un DataFrame llamado `df`.

El DataFrame `df` ya está cargado y contiene los datos procesados del historial de escucha.

El dataset cubre desde {fecha_min} hasta {fecha_max}.
Las plataformas disponibles son: {plataformas}.
Los valores posibles de reason_start son: {reason_start_values}.
Los valores posibles de reason_end son: {reason_end_values}.

Puedes responder únicamente preguntas analíticas basadas en los datos, incluyendo:

- Rankings (artistas, canciones, álbumes)
- Evolución temporal (por día, mes, año)
- Patrones de uso (horas, días de la semana, fin de semana vs entre semana)
- Comportamiento de escucha (skips, shuffle, duración)
- Comparaciones entre periodos (meses, años, estaciones)

Columnas disponibles en `df`:

- ts (datetime)
- date, year, month, month_name, day, hour
- day_of_week, day_name, is_weekend
- season
- artist, track, album
- minutes_played, hours_played
- skipped, shuffle
- platform, reason_start, reason_end

Reglas para el código:

- Usa únicamente pandas (pd), plotly.express (px) y plotly.graph_objects (go)
- El DataFrame disponible se llama `df`
- NO cargues archivos ni redefinas `df`
- NO inventes columnas ni uses campos que no estén en `df`
- NO uses print()
- NO uses st. (streamlit)
- NO uses matplotlib

- El código debe generar SIEMPRE una variable llamada `fig`
- `fig` debe ser una figura de Plotly (px o go)

Formato de salida:

Debes devolver SIEMPRE un único JSON válido con esta estructura:

{{
  "tipo": "grafico" o "fuera_de_alcance",
  "codigo": "código Python ejecutable",
  "interpretacion": "explicación breve del resultado"
}}

- NO incluyas texto fuera del JSON

Si la pregunta no puede resolverse con los datos disponibles:

- responde con:
{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "explica brevemente por qué no se puede responder"
}}
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    df = df.rename(columns={
        "master_metadata_track_name": "track",
        "master_metadata_album_artist_name": "artist",
        "master_metadata_album_album_name": "album"
    })

    # 1. Convertir timestamp a datetime
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    # 2. Eliminar filas sin timestamp válido
    df = df.dropna(subset=["ts"]).copy()

    # 3. Crear columnas temporales derivadas
    df["date"] = df["ts"].dt.date
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    df["month_name"] = df["ts"].dt.month_name()
    df["day"] = df["ts"].dt.day
    df["hour"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek
    df["day_name"] = df["ts"].dt.day_name()

    # 4. Crear indicador de fin de semana
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    # 5. Convertir duración a minutos y horas
    df["minutes_played"] = df["ms_played"] / 60000
    df["hours_played"] = df["ms_played"] / 3600000

    # 6. Estandarizar columnas booleanas si existen
    if "skipped" in df.columns:
        df["skipped"] = df["skipped"].fillna(False).astype(bool)

    if "shuffle" in df.columns:
        df["shuffle"] = df["shuffle"].fillna(False).astype(bool)

    # 7. Crear estación del año a partir del mes
    season_map = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn"
    }
    df["season"] = df["month"].map(season_map)

    # 8. Limpiar columnas de texto frecuentes si existen
    text_cols = ["artist", "track", "album", "platform", "reason_start", "reason_end"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # 9. Eliminar registros sin artista o canción identificable
    df = df[(df["artist"] != "Unknown") & (df["track"] != "Unknown")].copy()

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    Mi aplicación sigue una arquitectura text-to-code: el usuario escribe una pregunta en lenguaje natural y el LLM devuelve un JSON con código Python, tipo de respuesta e interpretación breve.
#    El modelo no calcula directamente el resultado, sino que genera código para analizar un DataFrame `df` ya cargado y preparado en la app.
#    Ese código se ejecuta después con `exec()` en Python, en un entorno controlado con acceso solo a `df`, `pd`, `px` y `go`.
#    El LLM no recibe los datos directamente porque así reduzco coste y contexto, y obligo a que el resultado salga de código ejecutado sobre datos reales, no de texto inventado.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    El system prompt define el comportamiento del modelo: qué preguntas puede resolver, qué columnas existen en `df`, qué librerías puede usar, que debe crear una figura Plotly llamada `fig` y que la salida debe ser un JSON válido.
#    También le paso información dinámica del dataset, como rango de fechas, plataformas y valores de `reason_start` y `reason_end`, para ajustar mejor las respuestas.
#    Por ejemplo, la pregunta “¿Más entre semana o el fin de semana?” funciona bien porque en `load_data()` creo la columna `is_weekend` y en el prompt indico que existe.
#    Si quitase la instrucción de no inventar columnas, el modelo podría usar nombres inexistentes y generar código que fallaría al ejecutarse.entar
#    columnas”, el modelo podría intentar usar nombres que no existen y generar código que fallaría al ejecutarse.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    Cuando el usuario escribe una pregunta, Streamlit la recoge y la envía junto con el system prompt a la API.
#    Antes de eso, la app ya ha cargado el JSON y ha preparado `df` con columnas derivadas como `hour`, `day_name`, `is_weekend`, `minutes_played` o `season`.
#    El LLM devuelve un texto que debería ser un JSON; la app lo parsea y comprueba el campo `tipo`.
#    Si es `fuera_de_alcance`, muestra solo la interpretación; si es `grafico`, ejecuta el código generado, obtiene `fig` y renderiza el gráfico en pantalla junto con la interpretación y el código.