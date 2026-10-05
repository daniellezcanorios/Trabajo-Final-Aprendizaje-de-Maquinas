import pandas as pd
import streamlit as st
import joblib


# ---------------------------------------------------------
# Configuración general
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predicción de riesgo académico",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def cargar_modelo():
    """Carga el pipeline serializado con preprocesamiento y modelo."""
    return joblib.load("modelo_final_saberpro_comprimido_v2.pkl")


try:
    modelo = cargar_modelo()
except FileNotFoundError:
    st.error(
        "No se encontró el archivo modelo_final_saberpro_comprimido.pkl. "
        "Verifica que esté ubicado en la misma carpeta que app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"No fue posible cargar el modelo: {error}")
    st.stop()


# ---------------------------------------------------------
# Encabezado
# ---------------------------------------------------------
st.title("🎓 Predicción de riesgo de bajo desempeño")
st.markdown(
    """
    ### Proyecto académico
    **Maestría en Ciencia de Datos · Universidad Pontificia Bolivariana (UPB)**

    Esta aplicación utiliza un modelo de Machine Learning para estimar el riesgo
    de bajo desempeño en razonamiento cuantitativo a partir de características
    académicas y socioeconómicas.
    """
)

st.info(
    "La predicción es una herramienta de apoyo preventivo. No debe utilizarse "
    "para tomar decisiones excluyentes sobre estudiantes."
)

st.divider()


# ---------------------------------------------------------
# Formulario interactivo
# ---------------------------------------------------------
st.subheader("Información del estudiante")

with st.form("formulario_prediccion"):
    columna_1, columna_2 = st.columns(2)

    with columna_1:
        periodo = st.selectbox(
            "Periodo de presentación",
            [
                "20183",
                "20184",
                "20194",
                "20195",
                "20203",
                "20204",
                "20212",
            ],
            help="Selecciona uno de los periodos representados en los datos de entrenamiento.",
        )

        modalidad = st.selectbox(
            "Modalidad del programa",
            [
                "PRESENCIAL",
                "DISTANCIA TRADICIONAL",
                "DISTANCIA VIRTUAL",
            ],
        )

        horas_trabajo = st.select_slider(
            "Horas de trabajo semanales",
            options=[
                "No trabaja",
                "Menos de 10 horas",
                "Entre 11 y 20 horas",
                "Entre 21 y 30 horas",
                "Más de 30 horas",
            ],
            value="Menos de 10 horas",
        )

        genero = st.radio(
            "Género registrado",
            ["F", "M"],
            horizontal=True,
        )

        estrato_numero = st.slider(
            "Estrato socioeconómico",
            min_value=1,
            max_value=6,
            value=2,
            step=1,
        )

    with columna_2:
        niveles_educativos = [
            "No sabe",
            "Ninguno",
            "Primaria incompleta",
            "Primaria completa",
            "Secundaria (Bachillerato) incompleta",
            "Secundaria (Bachillerato) completa",
            "Técnica o tecnológica incompleta",
            "Técnica o tecnológica completa",
            "Educación profesional incompleta",
            "Educación profesional completa",
            "Postgrado",
        ]

        educacion_padre = st.selectbox(
            "Nivel educativo del padre",
            niveles_educativos,
            index=5,
        )

        educacion_madre = st.selectbox(
            "Nivel educativo de la madre",
            niveles_educativos,
            index=5,
        )

        acceso_internet = st.toggle(
            "Cuenta con acceso a internet",
            value=True,
        )

        acceso_computador = st.toggle(
            "Cuenta con computador",
            value=True,
        )

        origen_institucion = st.selectbox(
            "Origen de la institución",
            [
                "OFICIAL MUNICIPAL",
                "OFICIAL DEPARTAMENTAL",
                "OFICIAL NACIONAL",
                "NO OFICIAL - FUNDACIÓN",
                "NO OFICIAL - CORPORACIÓN",
                "REGIMEN ESPECIAL",
            ],
            index=1,
        )

    enviar = st.form_submit_button(
        "🔍 Realizar predicción",
        use_container_width=True,
    )


# ---------------------------------------------------------
# Predicción
# ---------------------------------------------------------
if enviar:
    estrato = f"Estrato {estrato_numero}"
    tiene_internet = "Si" if acceso_internet else "No"
    tiene_computador = "Si" if acceso_computador else "No"

    entrada = pd.DataFrame(
        {
            "PERIODO": [periodo],
            "ESTU_METODO_PRGM": [modalidad],
            "ESTU_HORASSEMANATRABAJA": [horas_trabajo],
            "ESTU_GENERO": [genero],
            "FAMI_EDUCACIONPADRE": [educacion_padre],
            "FAMI_ESTRATOVIVIENDA": [estrato],
            "FAMI_TIENECOMPUTADOR": [tiene_computador],
            "FAMI_TIENEINTERNET": [tiene_internet],
            "FAMI_EDUCACIONMADRE": [educacion_madre],
            "INST_ORIGEN": [origen_institucion],
        }
    )

    try:
        prediccion = int(modelo.predict(entrada)[0])
        probabilidad_riesgo = float(modelo.predict_proba(entrada)[0][1])
    except Exception as error:
        st.error(f"No fue posible realizar la predicción: {error}")
        st.stop()

    st.divider()
    st.subheader("Resultado de la predicción")

    resultado_1, resultado_2 = st.columns([1, 2])

    with resultado_1:
        st.metric(
            "Probabilidad estimada de riesgo",
            f"{probabilidad_riesgo * 100:.2f}%",
        )
        st.progress(probabilidad_riesgo)

    with resultado_2:
        if prediccion == 1:
            st.error("⚠️ ESTUDIANTE CON RIESGO DE BAJO DESEMPEÑO")
            st.write(
                "El modelo identifica un perfil asociado con riesgo de bajo "
                "desempeño en razonamiento cuantitativo. Se recomienda validar "
                "el resultado y considerar acciones preventivas de acompañamiento académico."
            )
        else:
            st.success("✅ ESTUDIANTE SIN RIESGO DE BAJO DESEMPEÑO")
            st.write(
                "El modelo identifica un perfil asociado con desempeño esperado "
                "en razonamiento cuantitativo. El resultado no reemplaza el "
                "seguimiento académico individual."
            )

    with st.expander("Ver datos utilizados en la predicción"):
        st.dataframe(entrada, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# Pie de página
# ---------------------------------------------------------
st.divider()
st.caption(
    "Developed by Daniel Lezcano Ríos · "
    "Maestría en Ciencia de Datos · Universidad Pontificia Bolivariana (UPB)"
)
