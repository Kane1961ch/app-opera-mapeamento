import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

# ==========================================
# 0. CONFIGURAÇÃO E ESTADO
# ==========================================
st.set_page_config(page_title="OPERA - Process Control", layout="wide", page_icon="🏭")

if 'df_pi' not in st.session_state: st.session_state['df_pi'] = None
if 'df_limpo' not in st.session_state: st.session_state['df_limpo'] = None
if 'mapeamento' not in st.session_state: st.session_state['mapeamento'] = []


# ==========================================
# FUNÇÕES DE MOTOR (BACKEND)
# ==========================================
def limpar_dados(df, remover_zeros, remover_negativos, outlier_metodo, fator):
    df_res = df.copy()

    # 1. Garante que todas as colunas que podem ser números virem números de fato
    df_res = df_res.apply(pd.to_numeric, errors='ignore')
    
    # 2. Separa APENAS as colunas que são números (ignora textos para não dar erro matemático)
    cols_num = df_res.select_dtypes(include=[np.number]).columns
    
    if len(cols_num) > 0:
        df_calc = df_res[cols_num].copy()

        # Substitui zeros e negativos por NaN (vazio) de forma super segura
        if remover_zeros:
            df_calc = df_calc.mask(df_calc.abs() < 1e-5)
        if remover_negativos:
            df_calc = df_calc.mask(df_calc < 0)

        # Filtro de Outliers
        if outlier_metodo == "IQR":
            q1 = df_calc.quantile(0.25)
            q3 = df_calc.quantile(0.75)
            iqr = q3 - q1
            df_calc = df_calc.mask((df_calc < (q1 - fator * iqr)) | (df_calc > (q3 + fator * iqr)))
        elif outlier_metodo == "Z-Score":
            z_scores = (df_calc - df_calc.mean()) / df_calc.std()
            df_calc = df_calc.mask(z_scores.abs() > fator)

        # Devolve as colunas limpas para o dataframe principal
        df_res[cols_num] = df_calc

    # Preenchimento inteligente (Forward e Backward Fill) para tampar os buracos
    return df_res.ffill().bfill()

    return df_res.ffill().bfill()


def avaliar_formula(df, formula, tags):
    df_temp = df[tags].copy()
    try:
        resultado = df_temp.iloc[:, 0]
        if "/" in formula and len(tags) >= 2:
            resultado = df_temp.iloc[:, 0] / df_temp.iloc[:, 1].replace(0, np.nan)
        elif "+" in formula:
            resultado = df_temp.sum(axis=1)
        elif "-" in formula and len(tags) >= 2:
            resultado = df_temp.iloc[:, 0] - df_temp.iloc[:, 1]
        elif "*" in formula:
            resultado = df_temp.product(axis=1)
        return resultado
    except Exception:
        return None


# ==========================================
# MENU LATERAL
# ==========================================
st.sidebar.title("🏭 OPERA v2.0")
menu = st.sidebar.selectbox("Navegação:", [
    "📂 1. Carga e Auditoria",
    "🧹 2. Limpeza Heurística",
    "📝 3. Mapeamento de Indicadores",
    "📊 4. Dashboard de Processo",
    "🧪 5. Análise Avançada (PCA/T²)"
])
st.sidebar.divider()
st.sidebar.info("App de monitoramento estatístico multivariado desenvolvido para a equipe de engenharia.")

# ==========================================
# 1. CARGA E AUDITORIA
# ==========================================
if menu == "📂 1. Carga e Auditoria":
    st.title("📂 Carga e Auditoria de Sensores")
    arquivo = st.file_uploader("Arraste seu arquivo aqui (.pkl ou .xlsx)", type=["pkl", "xlsx"])

    if arquivo:
        with st.spinner("Processando base de dados..."):
            df = pd.read_pickle(arquivo) if arquivo.name.endswith('.pkl') else pd.read_excel(arquivo, index_col=0)
            st.session_state['df_pi'] = df
            st.success(f"✅ Base carregada: {df.shape[0]} instantes e {df.shape[1]} tags.")
            st.dataframe(df.head(3))

        st.divider()
        st.subheader("🕵️ Auditoria de Redundância (Sensores Clonados)")
        if st.button("Executar Auditoria de Clones"):
            with st.spinner("Cruzando correlações..."):
                corr = df.select_dtypes(include=[np.number]).corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                clones = [(col, row, upper.loc[row, col]) for col in upper.columns for row in upper.index if
                          upper.loc[row, col] > 0.99]

                if clones:
                    st.error(f"🚨 Detectados {len(clones)} pares de sensores redundantes!")
                    st.dataframe(pd.DataFrame(clones, columns=["Sensor A", "Sensor B", "Correlação"]).sort_values(
                        by="Correlação", ascending=False))
                else:
                    st.success("✅ Base saudável! Nenhum sensor clonado detectado.")

# ==========================================
# 2. LIMPEZA HEURÍSTICA
# ==========================================
elif menu == "🧹 2. Limpeza Heurística":
    st.title("🧹 Filtros de Engenharia")
    if st.session_state['df_pi'] is None:
        st.warning("⚠️ Carregue a base de dados no Módulo 1 primeiro.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            z = st.checkbox("Remover Zeros", value=True)
            n = st.checkbox("Remover Negativos", value=True)
        with col2:
            metodo = st.selectbox("Filtro de Outliers", ["Nenhum", "IQR", "Z-Score"])
            fator = st.number_input("Fator de Sensibilidade", value=1.5 if metodo == "IQR" else 3.0, step=0.1)

        if st.button("🧹 Aplicar Limpeza"):
            with st.spinner("Limpando histórico temporal..."):
                df_limpo = limpar_dados(st.session_state['df_pi'], z, n, metodo, fator)
                st.session_state['df_limpo'] = df_limpo
                st.success("✅ Dados limpos e prontos!")

                fig, ax = plt.subplots(figsize=(10, 3))
                primeira_tag = df_limpo.columns[0]
                ax.plot(st.session_state['df_pi'].index, st.session_state['df_pi'][primeira_tag], alpha=0.3,
                        label="Original", color='red')
                ax.plot(df_limpo.index, df_limpo[primeira_tag], alpha=0.8, label="Limpo", color='blue')
                ax.legend()
                st.pyplot(fig)

# ==========================================
# 3. MAPEAMENTO DE INDICADORES
# ==========================================
elif menu == "📝 3. Mapeamento de Indicadores":
    st.title("📝 Configuração de Sensores Virtuais")
    if st.session_state['df_pi'] is None:
        st.warning("⚠️ Carregue a base de dados no Módulo 1 primeiro.")
    else:
        with st.form("form_indicador"):
            nome_ind = st.text_input("Nome do Indicador")
            formula = st.selectbox("Operação Matemática Principal",
                                   ["Divisão (/)", "Multiplicação (*)", "Soma (+)", "Subtração (-)"])
            tags_selecionadas = st.multiselect("Selecione as Tags Reais", st.session_state['df_pi'].columns)

            if st.form_submit_button("💾 Salvar Indicador") and len(tags_selecionadas) > 0:
                st.session_state['mapeamento'].append({"nome": nome_ind, "formula": formula, "tags": tags_selecionadas})
                st.success(f"Indicador '{nome_ind}' salvo!")

        if st.session_state['mapeamento']:
            st.table(pd.DataFrame(st.session_state['mapeamento']))
            if st.button("Limpar Todos os Indicadores"):
                st.session_state['mapeamento'] = []
                st.rerun()

# ==========================================
# 4. DASHBOARD DE PROCESSO
# ==========================================
elif menu == "📊 4. Dashboard de Processo":
    st.title("📊 Monitoramento de Performance")
    if not st.session_state['mapeamento']:
        st.info("ℹ️ Nenhum indicador mapeado. Vá ao Módulo 3.")
    else:
        df_target = st.session_state['df_limpo'] if st.session_state['df_limpo'] is not None else st.session_state[
            'df_pi']
        for item in st.session_state['mapeamento']:
            with st.expander(f"📈 Analisar: {item['nome']}", expanded=True):
                serie_calc = avaliar_formula(df_target, item['formula'], item['tags'])
                if serie_calc is not None and not serie_calc.isna().all():
                    media, desvio = serie_calc.mean(), serie_calc.std()
                    colA, colB, colC = st.columns(3)
                    colA.metric("Média", f"{media:.2f}")
                    colB.metric("Máximo", f"{serie_calc.max():.2f}")
                    colC.metric("Mínimo", f"{serie_calc.min():.2f}")

                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(serie_calc.index, serie_calc.values, color='#1f77b4')
                    ax.axhline(media, color='green', label=f'Média ({media:.2f})')
                    ax.axhline(media + 3 * desvio, color='red', linestyle='--', label='LSC')
                    ax.axhline(media - 3 * desvio, color='red', linestyle='--', label='LIC')
                    ax.legend()
                    st.pyplot(fig)

# ==========================================
# 5. ANÁLISE AVANÇADA (PCA E HOTELLING)
# ==========================================
elif menu == "🧪 5. Análise Avançada (PCA/T²)":
    st.title("🧪 Diagnóstico Avançado")
    if st.session_state['df_limpo'] is None:
        st.warning("⚠️ O T² e o PCA exigem matrizes limpas. Execute o Módulo 2 primeiro.")
    else:
        df_target = st.session_state['df_limpo']
        tab1, tab2 = st.tabs(["📉 PCA", "🎯 Carta T²"])

        with tab1:
            termo_pca = st.text_input("Filtrar tags para o PCA:", key="pca_input")
            if st.button("🚀 Processar PCA") and termo_pca:
                cols_encontradas = [c for c in df_target.columns if termo_pca.lower() in str(c).lower()]
                if len(cols_encontradas) > 1:
                    df_pca = df_target[cols_encontradas].dropna()
                    df_pca = df_pca.loc[:, df_pca.std() > 1e-6]
                    pca_data = PCA().fit_transform(StandardScaler().fit_transform(df_pca))

                    fig, ax = plt.subplots(figsize=(8, 4))
                    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=range(len(pca_data)), cmap='viridis',
                                         alpha=0.5)
                    ax.set_title("Mapa de Estado Operacional (PC1 vs PC2)")
                    plt.colorbar(scatter, label='Evolução Temporal')
                    st.pyplot(fig)

        with tab2:
            termo_t2 = st.text_input("Agrupar equipamentos para o T²:", key="t2_input")
            confianca = st.slider("Confiança", 0.90, 0.999, 0.99)
            if st.button("🚀 Calcular Carta T²") and termo_t2:
                cols_encontradas = [c for c in df_target.columns if termo_t2.lower() in str(c).lower()]
                if len(cols_encontradas) > 1:
                    try:
                        df_t2 = df_target[cols_encontradas].dropna()
                        df_t2 = df_t2.loc[:, df_t2.std() > 1e-6]
                        diff = df_t2 - df_t2.mean()
                        t2_valores = np.sum((np.dot(diff, np.linalg.inv(df_t2.cov()))) * diff, axis=1)
                        lsc_t2 = chi2.ppf(confianca, df=df_t2.shape[1])

                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(range(len(t2_valores)), t2_valores, color='purple')
                        ax.axhline(lsc_t2, color='red', label='Limite T²')
                        ax.legend()
                        st.pyplot(fig)
                    except np.linalg.LinAlgError:
                        st.error("❌ ERRO: Matriz singular. Remova sensores clonados no Módulo 1.")
