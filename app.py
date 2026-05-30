import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import pickle
from google.cloud import storage
from river import linear_model, preprocessing, metrics, optim

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Aprendizaje en línea", page_icon="🚕")
st.title("Aprendizaje en línea con River — GCS Step-by-step")

st.markdown("""
Entrena un modelo de **aprendizaje incremental** procesando
**un archivo por clic** desde Google Cloud Storage.

El estado (índice, historial, modelo) se guarda en GCS para que
persista entre reinicios del contenedor.
""")

# =========================================================
# GCS HELPERS
# =========================================================

def _client():
    return storage.Client()


def save_model_to_gcs(model, bucket_name, blob_path):
    try:
        blob = _client().bucket(bucket_name).blob(blob_path)
        blob.upload_from_string(pickle.dumps(model))
        st.success(f"Modelo guardado: `gs://{bucket_name}/{blob_path}`")
    except Exception as e:
        st.warning(f"No se pudo guardar el modelo: {e}")


def load_model_from_gcs(bucket_name, blob_path):
    try:
        blob = _client().bucket(bucket_name).blob(blob_path)
        if blob.exists():
            return pickle.loads(blob.download_as_bytes())
        return None
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo: {e}")
        return None


def save_state_to_gcs(state_dict, bucket_name, blob_path):
    """Persiste index + historial como JSON para sobrevivir reinicios del contenedor."""
    try:
        blob = _client().bucket(bucket_name).blob(blob_path)
        blob.upload_from_string(json.dumps(state_dict).encode("utf-8"))
    except Exception as e:
        st.warning(f"No se pudo guardar el estado: {e}")


def load_state_from_gcs(bucket_name, blob_path):
    try:
        blob = _client().bucket(bucket_name).blob(blob_path)
        if blob.exists():
            return json.loads(blob.download_as_bytes().decode("utf-8"))
        return None
    except Exception as e:
        st.warning(f"No se pudo cargar el estado: {e}")
        return None


def delete_blob(bucket_name, blob_path):
    try:
        blob = _client().bucket(bucket_name).blob(blob_path)
        if blob.exists():
            blob.delete()
    except Exception as e:
        st.warning(f"No se pudo eliminar `{blob_path}`: {e}")


# =========================================================
# MODEL FACTORY
# =========================================================

def new_model():
    return preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.SGD(0.001),
        intercept_lr=0.001,
    )


# =========================================================
# SIDEBAR — PARAMETERS
# =========================================================

with st.sidebar:
    st.header("Configuración")
    data_bucket  = st.text_input("Bucket de datos (CSV):", "nyc_taxi_leo")
    prefix       = st.text_input("Prefijo/carpeta:", "")
    model_bucket = st.text_input("Bucket del modelo:", "actividad-3-leonel")
    limite       = st.number_input("Filas a procesar por archivo:", value=1000, step=100)
    st.markdown("---")

    if st.button("Reiniciar entrenamiento"):
        delete_blob(model_bucket, "models/model_incremental.pkl")
        delete_blob(model_bucket, "state/app_state.json")

        for k, v in [
            ("model",             new_model()),
            ("metric_r2",         metrics.R2()),
            ("metric_mae",        metrics.MAE()),
            ("history_r2",        []),
            ("history_mae",       []),
            ("history_file_r2",   []),
            ("history_file_mae",  []),
            ("processed_files",   []),
            ("index",             0),
            ("blobs",             None),
            ("loaded",            True),
        ]:
            st.session_state[k] = v

        st.success("Entrenamiento reiniciado.")

MODEL_PATH = "models/model_incremental.pkl"
STATE_PATH = "state/app_state.json"

# =========================================================
# INIT SESSION STATE
# La primera vez (o tras reinicio del contenedor) se carga
# el modelo y el estado desde GCS en lugar de memoria local.
# =========================================================

if "loaded" not in st.session_state:

    loaded_model = load_model_from_gcs(model_bucket, MODEL_PATH)
    loaded_state = load_state_from_gcs(model_bucket, STATE_PATH)

    st.session_state.model      = loaded_model if loaded_model is not None else new_model()
    st.session_state.metric_r2  = metrics.R2()
    st.session_state.metric_mae = metrics.MAE()

    if loaded_state is not None:
        st.session_state.history_r2       = loaded_state.get("history_r2",       [])
        st.session_state.history_mae      = loaded_state.get("history_mae",      [])
        st.session_state.history_file_r2  = loaded_state.get("history_file_r2",  [])
        st.session_state.history_file_mae = loaded_state.get("history_file_mae", [])
        st.session_state.processed_files  = loaded_state.get("processed_files",  [])
        st.session_state.index            = loaded_state.get("index",            0)
        st.info(f"Estado restaurado desde GCS. Archivos previos: {st.session_state.index}")
    else:
        st.session_state.history_r2       = []
        st.session_state.history_mae      = []
        st.session_state.history_file_r2  = []
        st.session_state.history_file_mae = []
        st.session_state.processed_files  = []
        st.session_state.index            = 0

    st.session_state.blobs  = None
    st.session_state.loaded = True

model      = st.session_state.model
metric_r2  = st.session_state.metric_r2
metric_mae = st.session_state.metric_mae


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def _parse_time_fields(row):
    if "pickup_hour" in row and pd.notna(row["pickup_hour"]):
        try:
            hour = int(pd.to_numeric(row["pickup_hour"], errors="coerce"))
            return None, max(0, min(hour, 23))
        except Exception:
            pass
    for c in ("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"):
        if c in row and pd.notna(row[c]):
            dt = pd.to_datetime(row[c], errors="coerce", utc=False)
            if pd.notna(dt):
                return dt, int(dt.hour)
    return None, 0


def _extract_x(row):
    dist    = float(row["trip_distance"])
    psg     = float(row["passenger_count"])
    dt, hour = _parse_time_fields(row)
    dow     = int(dt.weekday()) if isinstance(dt, pd.Timestamp) else 0
    weekend = 1.0 if dow >= 5 else 0.0
    return {
        "dist":       dist,
        "log_dist":   float(np.log1p(max(dist, 0.0))),
        "pass":       psg,
        "hour":       float(hour),
        "dow":        float(dow),
        "is_weekend": weekend,
    }


# =========================================================
# PROCESS ONE FILE
# =========================================================

def process_single_blob(data_bkt, blob_name, limite=1000, chunksize=500):
    blob   = _client().bucket(data_bkt).blob(blob_name)
    chunks = []

    try:
        buffer = io.BytesIO(blob.download_as_bytes())
        for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):
            required = ["trip_distance", "passenger_count", "fare_amount"]
            if not set(required).issubset(chunk.columns):
                continue
            for col in required:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
            chunk = chunk[
                chunk["fare_amount"].between(2, 200) &
                chunk["trip_distance"].between(0.1, 50) &
                chunk["passenger_count"].between(1, 6)
            ]
            if not chunk.empty:
                chunks.append(chunk)
    except Exception as e:
        st.warning(f"Error leyendo `{blob_name}`: {e}")
        return None

    if not chunks:
        return None

    df_file = pd.concat(chunks, ignore_index=True)
    if len(df_file) > limite:
        df_file = df_file.sample(n=limite, random_state=42)

    file_r2  = metrics.R2()
    file_mae = metrics.MAE()
    count    = 0

    for _, row in df_file.iterrows():
        y         = float(row["fare_amount"])
        x         = _extract_x(row)
        pred      = model.predict_one(x)
        pred_eval = float(np.clip(pred, 2, 200)) if pred is not None else 0.0

        metric_r2.update(y, pred_eval)
        metric_mae.update(y, pred_eval)
        file_r2.update(y, pred_eval)
        file_mae.update(y, pred_eval)

        model.learn_one(x, y)
        count += 1

    return {
        "count":      count,
        "file_r2":    file_r2.get(),
        "file_mae":   file_mae.get(),
        "global_r2":  metric_r2.get(),
        "global_mae": metric_mae.get(),
    }


# =========================================================
# MAIN — PROCESS NEXT FILE BUTTON
# =========================================================

st.markdown("---")
st.subheader("Procesamiento incremental")

if st.button("▶ Procesar siguiente archivo"):

    if st.session_state.blobs is None:
        bkt   = _client().bucket(data_bucket)
        blobs = [
            b for b in bkt.list_blobs(prefix=prefix)
            if b.name.endswith(".csv") and not b.name.endswith("/")
        ]
        st.session_state.blobs = blobs
        st.info(f"Se encontraron {len(blobs)} archivos CSV en `gs://{data_bucket}/{prefix}`.")

    blobs = st.session_state.blobs
    idx   = st.session_state.index

    if idx >= len(blobs):
        st.success("Todos los archivos ya fueron procesados.")
    else:
        blob  = blobs[idx]
        short = blob.name.split("/")[-1]

        with st.spinner(f"Procesando {short}..."):
            result = process_single_blob(
                data_bkt=data_bucket,
                blob_name=blob.name,
                limite=int(limite),
            )

        st.write(f"**Archivo {idx + 1}/{len(blobs)}:** `{short}`")

        if result is not None:
            st.session_state.history_r2.append(result["global_r2"])
            st.session_state.history_mae.append(result["global_mae"])
            st.session_state.history_file_r2.append(result["file_r2"])
            st.session_state.history_file_mae.append(result["file_mae"])
            st.session_state.processed_files.append(short)

            col1, col2, col3 = st.columns(3)
            col1.metric("Registros procesados",  result["count"])
            col2.metric("R² archivo",            f"{result['file_r2']:.4f}")
            col3.metric("MAE archivo",           f"{result['file_mae']:.4f}")

            col4, col5 = st.columns(2)
            col4.metric("R² acumulado",  f"{result['global_r2']:.4f}")
            col5.metric("MAE acumulado", f"{result['global_mae']:.4f}")

            # Persistir modelo y estado en GCS para sobrevivir reinicios
            save_model_to_gcs(model, model_bucket, MODEL_PATH)
            save_state_to_gcs(
                {
                    "index":            idx + 1,
                    "history_r2":       st.session_state.history_r2,
                    "history_mae":      st.session_state.history_mae,
                    "history_file_r2":  st.session_state.history_file_r2,
                    "history_file_mae": st.session_state.history_file_mae,
                    "processed_files":  st.session_state.processed_files,
                },
                model_bucket,
                STATE_PATH,
            )
        else:
            st.warning("No se procesaron registros válidos en este archivo.")

        st.session_state.index += 1


# =========================================================
# STATUS
# =========================================================

st.markdown("---")
st.subheader("Estado actual")

col1, col2, col3 = st.columns(3)
col1.metric("Archivos procesados", st.session_state.index)
col2.metric("R² acumulado",        f"{metric_r2.get():.4f}")
col3.metric("MAE acumulado",       f"{metric_mae.get():.4f}")

# =========================================================
# HISTORY
# =========================================================

if st.session_state.history_r2:
    df_hist = pd.DataFrame({
        "archivo":       st.session_state.processed_files,
        "R2_archivo":    st.session_state.history_file_r2,
        "MAE_archivo":   st.session_state.history_file_mae,
        "R2_acumulado":  st.session_state.history_r2,
        "MAE_acumulado": st.session_state.history_mae,
    })

    st.subheader("Historial de procesamiento")
    st.dataframe(df_hist, use_container_width=True)

    st.subheader("Evolución de métricas acumuladas")
    st.line_chart(df_hist[["R2_acumulado", "MAE_acumulado"]])

    st.subheader("Métricas por archivo")
    st.line_chart(df_hist[["R2_archivo", "MAE_archivo"]])

st.caption("Cloud Run · River · GCS — Dataset NYC Taxis 2022 · Leonel García Melena")
