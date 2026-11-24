import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import streamlit as st

st.set_page_config(
    page_title="Optimización Inferencial del Rendimiento Agrícola",
    page_icon="🌱",
    layout="wide"
)


@st.cache_data
def cargar_telemetria_duckdb(semilla: float = 0.42, n_parcelas: int = 100) -> pd.DataFrame:
    """Hito 1: Genera telemetría determinista de parcelas en DuckDB y la extrae a pandas."""
    con = duckdb.connect(database=":memory:")
    con.execute(f"SELECT setseed({semilla});")

    sql_query = f"""
    WITH seq AS (
        SELECT generate_series AS parcela_id
        FROM generate_series(1, {n_parcelas})
    ),
    simulacion_base AS (
        SELECT
            parcela_id,
            ROUND(100.0 + random() * (300.0 - 100.0), 2) AS volumen_riego,
            ROUND(50.0 + random() * (150.0 - 50.0), 2) AS fertilizante_kg,
            random() AS prob_tratamiento,
            -0.8 + random() * (0.8 - (-0.8)) AS ruido_aleatorio,
            ROUND(18.0 + random() * (32.0 - 18.0), 2) AS temperatura_c
        FROM seq
    ),
    con_tratamiento AS (
        SELECT
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            temperatura_c,
            CASE
                WHEN prob_tratamiento < 0.50 THEN 'Tratamiento_A'
                ELSE 'Tratamiento_B'
            END AS tipo_tratamiento,
            CASE
                WHEN prob_tratamiento < 0.50 THEN 0.0
                ELSE 1.8
            END AS bono_tratamiento,
            ruido_aleatorio
        FROM simulacion_base
    ),
    telemetria_limpia AS (
        SELECT
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            tipo_tratamiento,
            temperatura_c,
            ROUND(
                5.0
                + (0.035 * volumen_riego)
                + (0.020 * fertilizante_kg)
                + bono_tratamiento
                + ruido_aleatorio,
                2
            ) AS rendimiento_ton
        FROM con_tratamiento
        WHERE volumen_riego IS NOT NULL
          AND fertilizante_kg IS NOT NULL
          AND rendimiento_ton IS NOT NULL
    )
    SELECT * FROM telemetria_limpia ORDER BY parcela_id;
    """
    df_sim = con.execute(sql_query).df()
    con.close()
    return df_sim


def ajustar_ols(data: pd.DataFrame):
    """Hito 2: Ajuste del modelo de Regresión Lineal Múltiple OLS."""
    X = sm.add_constant(data[["volumen_riego", "fertilizante_kg"]])
    y = data["rendimiento_ton"]
    modelo = sm.OLS(y, X).fit()
    y_pred = modelo.predict(X)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    residuos = y - y_pred
    return modelo, y_pred, rmse, residuos


def prueba_hipotesis_tratamientos(data: pd.DataFrame):
    """Hito 3: Prueba t-Student de dos muestras independientes (H1: mu_B > mu_A)."""
    grupo_a = data[data["tipo_tratamiento"] == "Tratamiento_A"]["rendimiento_ton"].dropna()
    grupo_b = data[data["tipo_tratamiento"] == "Tratamiento_B"]["rendimiento_ton"].dropna()

    if len(grupo_a) < 2 or len(grupo_b) < 2:
        return None

    t_stat, p_val = stats.ttest_ind(grupo_b, grupo_a, equal_var=False, alternative="greater")
    diff_medias = float(grupo_b.mean() - grupo_a.mean())

    var_a, var_b = grupo_a.var(ddof=1), grupo_b.var(ddof=1)
    n_a, n_b = len(grupo_a), len(grupo_b)
    se = np.sqrt((var_b / n_b) + (var_a / n_a))

    if se > 0:
        gl = ((var_b / n_b) + (var_a / n_a)) ** 2 / (
            ((var_b / n_b) ** 2) / (n_b - 1) + ((var_a / n_a) ** 2) / (n_a - 1)
        )
        t_crit = stats.t.ppf(0.975, df=gl)
        ic_inf = float(diff_medias - t_crit * se)
        ic_sup = float(diff_medias + t_crit * se)
    else:
        ic_inf, ic_sup = diff_medias, diff_medias

    return {
        "n_a": n_a,
        "n_b": n_b,
        "media_a": float(grupo_a.mean()),
        "media_b": float(grupo_b.mean()),
        "diff_medias": diff_medias,
        "ic_95": (ic_inf, ic_sup),
        "t_stat": float(t_stat),
        "p_val": float(p_val),
    }


def main():
    st.title("🌱 Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego")
    st.caption(
        "Proyecto Integrador U4 — Análisis y Exploración de Datos (INC-2505) | "
        "Telemetría DuckDB, Regresión OLS e Inferencia Estadística"
    )

    df_full = cargar_telemetria_duckdb(semilla=0.42, n_parcelas=100)

    # Barra lateral: filtros dinámicos (Hito 4)
    st.sidebar.header("🎛️ Filtros Interactivos")
    tratamientos_sel = st.sidebar.multiselect(
        "Tipo de Tratamiento",
        options=["Tratamiento_A", "Tratamiento_B"],
        default=["Tratamiento_A", "Tratamiento_B"],
    )
    riego_min, riego_max = st.sidebar.slider(
        "Volumen de Riego (m³/ha)",
        min_value=float(df_full["volumen_riego"].min()),
        max_value=float(df_full["volumen_riego"].max()),
        value=(float(df_full["volumen_riego"].min()), float(df_full["volumen_riego"].max())),
    )
    fert_min, fert_max = st.sidebar.slider(
        "Fertilizante Aplicado (kg/ha)",
        min_value=float(df_full["fertilizante_kg"].min()),
        max_value=float(df_full["fertilizante_kg"].max()),
        value=(float(df_full["fertilizante_kg"].min()), float(df_full["fertilizante_kg"].max())),
    )

    df_filtrado = df_full[
        (df_full["tipo_tratamiento"].isin(tratamientos_sel))
        & (df_full["volumen_riego"].between(riego_min, riego_max))
        & (df_full["fertilizante_kg"].between(fert_min, fert_max))
    ].copy()

    if len(df_filtrado) < 5:
        st.warning("Selecciona más registros o amplía los rangos de filtro para ajustar modelos y gráficas.")
        return

    modelo, y_pred, rmse, residuos = ajustar_ols(df_filtrado)
    prueba_full = prueba_hipotesis_tratamientos(df_full)
    prueba_filt = prueba_hipotesis_tratamientos(df_filtrado)

    # 3 Tarjetas KPI superiores (Fase 5)
    rend_promedio = float(df_filtrado["rendimiento_ton"].mean())
    r2_ajustado = float(modelo.rsquared_adj)
    p_valor_mostrar = prueba_filt["p_val"] if prueba_filt else (prueba_full["p_val"] if prueba_full else np.nan)

    k1, k2, k3 = st.columns(3)
    k1.metric("Rendimiento Promedio Global", f"{rend_promedio:.2f} ton/ha", f"N = {len(df_filtrado)} parcelas")
    k2.metric("R² Ajustado del Modelo (OLS)", f"{r2_ajustado:.4f} ({r2_ajustado*100:.2f}%)", f"RMSE: {rmse:.3f} ton/ha")
    p_str = f"{p_valor_mostrar:.4e}" if not np.isnan(p_valor_mostrar) else "N/A"
    decision_str = (
        "Rechaza H0 (α=0.05)"
        if (not np.isnan(p_valor_mostrar) and p_valor_mostrar < 0.05)
        else "No rechaza H0 / Filtro parcial"
    )
    k3.metric("P-Valor Prueba Inferencial (t)", p_str, decision_str)

    st.divider()

    # Cuadrícula 2x2 de gráficas (Fase 4 y Fase 5)
    sns.set_theme(style="whitegrid")

    fila1_col1, fila1_col2 = st.columns(2)
    fila2_col1, fila2_col2 = st.columns(2)

    # Gráfica 1: Heatmap de correlación de Pearson
    with fila1_col1:
        st.subheader("1. Mapa de Calor de Correlación (Pearson)")
        fig1, ax1 = plt.subplots(figsize=(6, 4.2))
        cols_corr = ["volumen_riego", "fertilizante_kg", "temperatura_c", "rendimiento_ton"]
        corr_df = df_filtrado[cols_corr].rename(columns={
            "volumen_riego": "Riego",
            "fertilizante_kg": "Fertilizante",
            "temperatura_c": "Temperatura",
            "rendimiento_ton": "Rendimiento"
        }).corr(method="pearson")
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=-1, vmax=1, ax=ax1)
        ax1.set_title("Matriz de Correlación Lineal (r)")
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    # Gráfica 2: Dispersión con línea OLS y banda de confianza
    with fila1_col2:
        st.subheader("2. Dispersión y Regresión OLS (Riego vs. Rendimiento)")
        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        sns.regplot(
            data=df_filtrado,
            x="volumen_riego",
            y="rendimiento_ton",
            scatter_kws={"alpha": 0.65, "s": 35, "color": "#2b6cb0"},
            line_kws={"color": "#c53030", "linewidth": 2},
            ci=95,
            ax=ax2
        )
        ax2.set_xlabel("Volumen de Riego (m³/ha)")
        ax2.set_ylabel("Rendimiento (ton/ha)")
        ax2.set_title("Ajuste Lineal con Banda de Confianza (95%)")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # Gráfica 3: Boxplot comparativo de tratamientos
    with fila2_col1:
        st.subheader("3. Boxplot Comparativo por Tratamiento")
        fig3, ax3 = plt.subplots(figsize=(6, 4.2))
        sns.boxplot(
            data=df_filtrado,
            x="tipo_tratamiento",
            y="rendimiento_ton",
            hue="tipo_tratamiento",
            palette=["#68d391", "#4299e1"],
            legend=False,
            ax=ax3
        )
        medias_grp = df_filtrado.groupby("tipo_tratamiento")["rendimiento_ton"].mean()
        for idx, trt in enumerate(medias_grp.index):
            ax3.scatter(
                idx,
                medias_grp.loc[trt],
                color="black",
                marker="D",
                s=45,
                zorder=5,
                label="Media" if idx == 0 else ""
            )
        ax3.set_xlabel("Tipo de Tratamiento")
        ax3.set_ylabel("Rendimiento (ton/ha)")
        ax3.set_title("Distribución de Rendimiento: Tratamiento_A vs. Tratamiento_B")
        if len(medias_grp) > 0:
            ax3.legend(loc="best")
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    # Gráfica 4: Histograma + KDE de residuos (Y_real - Y_predicho)
    with fila2_col2:
        st.subheader("4. Histograma y Densidad KDE de Residuos")
        fig4, ax4 = plt.subplots(figsize=(6, 4.2))
        sns.histplot(residuos, kde=True, bins=15, color="#319795", edgecolor="white", ax=ax4)
        ax4.axvline(0.0, color="#e53e3e", linestyle="--", linewidth=1.5, label="Residuo = 0")
        ax4.set_xlabel("Residuo (Y_real - Y_predicho) [ton/ha]")
        ax4.set_ylabel("Frecuencia / Densidad")
        ax4.set_title("Diagnóstico de Normalidad de Errores")
        ax4.legend(loc="best")
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

    st.divider()

    # Resumen estadístico detallado
    tab1, tab2, tab3 = st.tabs(["📊 Modelo OLS", "🧪 Prueba de Hipótesis (t)", "🗃️ Datos de Telemetría"])

    with tab1:
        st.write(
            f"**Ecuación ajustada:** $\\widehat{{\\text{{Rendimiento}}}} = {modelo.params['const']:.4f} "
            f"+ ({modelo.params['volumen_riego']:.4f} \\times \\text{{Riego}}) "
            f"+ ({modelo.params['fertilizante_kg']:.4f} \\times \\text{{Fertilizante}})$"
        )
        coef_df = pd.DataFrame({
            "Coeficiente": modelo.params,
            "Error Estándar": modelo.bse,
            "Estadístico t": modelo.tvalues,
            "P-Valor": modelo.pvalues,
        })
        st.dataframe(coef_df.round(6), use_container_width=True)
        st.caption(f"R²: {modelo.rsquared:.4f} | R² Ajustado: {modelo.rsquared_adj:.4f} | RMSE: {rmse:.4f} ton/ha")

    with tab2:
        if prueba_full:
            st.markdown(
                "- **Hipótesis Nula ($H_0$):** $\\mu_B - \\mu_A \\le 0$ (el nuevo bio-estimulante no incrementa el rendimiento medio).\n"
                "- **Hipótesis Alternativa ($H_1$):** $\\mu_B - \\mu_A > 0$ (el `Tratamiento_B` incrementa significativamente el rendimiento)."
            )
            res_inf = prueba_filt if prueba_filt else prueba_full
            c1, c2, c3 = st.columns(3)
            c1.metric("Media Tratamiento_A", f"{res_inf['media_a']:.3f} ton/ha", f"n = {res_inf['n_a']}")
            c2.metric("Media Tratamiento_B", f"{res_inf['media_b']:.3f} ton/ha", f"n = {res_inf['n_b']}")
            c3.metric(
                "Diferencia (B - A)",
                f"{res_inf['diff_medias']:+.3f} ton/ha",
                f"IC 95%: [{res_inf['ic_95'][0]:.3f}, {res_inf['ic_95'][1]:.3f}]"
            )

            if res_inf["p_val"] < 0.05:
                st.success(
                    f"Veredicto (α = 0.05): Se rechaza H₀ (t = {res_inf['t_stat']:.4f}, p = {res_inf['p_val']:.4e}). "
                    "Existe evidencia estadística suficiente para respaldar la adopción del Tratamiento_B, "
                    "sujeto a validación de costos unitarios del bio-estimulante frente al margen adicional obtenido."
                )
            else:
                st.info(
                    f"Veredicto (α = 0.05): No se rechaza H₀ (t = {res_inf['t_stat']:.4f}, p = {res_inf['p_val']:.4e}). "
                    "Con los datos actuales no hay evidencia suficiente para escalar el Tratamiento_B."
                )
        else:
            st.info("Incluye ambos tratamientos (Tratamiento_A y Tratamiento_B) con al menos 2 parcelas cada uno para ejecutar la prueba t.")

    with tab3:
        st.dataframe(df_filtrado, use_container_width=True)
        st.download_button(
            label="Descargar CSV de telemetría",
            data=df_filtrado.to_csv(index=False).encode("utf-8"),
            file_name="telemetria_invernadero.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()