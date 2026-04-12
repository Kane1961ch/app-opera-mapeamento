import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
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
if 'limites_customizados' not in st.session_state: st.session_state['limites_customizados'] = []

# ==========================================
# FUNÇÕES DE MOTOR (BACKEND)
# ==========================================
def limpar_dados(df, remover_zeros, remover_negativos, outlier_metodo, fator, limites_custom):
    df_res = df.copy()
    df_res = df_res.apply(pd.to_numeric, errors='coerce')
    
    # 1. Limites Específicos (Parte A)
    for regra in limites_custom:
        for tag in regra['tags']:
            if tag in df_res.columns:
                if regra['minimo'] is not None:
                    df_res[tag] = df_res[tag].mask(df_res[tag] < regra['minimo'])
                if regra['maximo'] is not None:
                    df_res[tag] = df_res[tag].mask(df_res[tag] > regra['maximo'])

    # 2. Filtros Globais (Parte B)
    if remover_zeros: df_res = df_res.mask(df_res.abs() < 1e-5)
    if remover_negativos: df_res = df_res.mask(df_res < 0)

    if outlier_metodo == "IQR":
        q1, q3 = df_res.quantile(0.25), df_res.quantile(0.75)
        iqr = q3 - q1
        df_res = df_res.mask((df_res < (q1 - fator * iqr)) | (df_res > (q3 + fator * iqr)))
    elif outlier_metodo == "Z-Score":
        z_scores = (df_res - df_res.mean()) / df_res.std()
        df_res = df_res.mask(z_scores.abs() > fator)

    return df_res.ffill().bfill()

def avaliar_formula_complexa(df, formula, tags):
    """Lê a string da fórmula e substitui pelas tags correspondentes"""
    termos_ignorar = {'SOMA', 'MEDIA', 'SE', 'MAX', 'MIN', 'IF', 'AND', 'OR'}
    palavras_brutas = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*%?', formula)
    vars_esperadas = list(dict.fromkeys([p for p in palavras_brutas if p.upper() not in termos_ignorar]))
    
    # Se não tem fórmula matemática explícita, soma as tags como padrão de segurança
    if not vars_esperadas:
        return df[tags].sum(axis=1)

    df_eval = pd.DataFrame()
    expressao = formula
    
    # Mapeia as variáveis da fórmula para as tags selecionadas na ordem
    for i, var in enumerate(vars_esperadas):
        if i < len(tags):
            safe_var = re.sub(r'[^a-zA-Z0-9_]', '', var)
            if not safe_var: safe_var = f"VAR_{i}"
            df_eval[safe_var] = df[tags[i]]
            expressao = re.sub(fr'\b{re.escape(var)}\b', safe_var, expressao)

    try:
        serie_ind = df_eval.eval(expressao, engine='python')
        return serie_ind.replace([np.inf, -np.inf], np.nan)
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
st.sidebar.info("App de monitoramento estatístico multivariado.")

# ==========================================
# 1. CARGA E AUDITORIA
# ==========================================
if menu == "📂 1. Carga e Auditoria":
    st.title("📂 Carga e Auditoria de Sensores")
    arquivo = st.file_uploader("Arraste seu arquivo (.pkl ou .xlsx)", type=["pkl", "xlsx"])

    if arquivo:
        with st.spinner("Processando base de dados..."):
            df = pd.read_pickle(arquivo) if arquivo.name.endswith('.pkl') else pd.read_excel(arquivo, index_col=0)
            st.session_state['df_pi'] = df
            st.success(f"✅ Base carregada: {df.shape[0]} instantes e {df.shape[1]} tags.")
            
        st.divider()
        st.subheader("🕵️ Auditoria de Redundância (Sensores Clonados)")
        if st.button("Executar Auditoria de Clones"):
            with st.spinner("Cruzando correlações..."):
                df_num = df.select_dtypes(include=[np.number])
                corr = df_num.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                clones = [(col, row, upper.loc[row, col]) for col in upper.columns for row in upper.index if upper.loc[row, col] > 0.99]

                if clones:
                    st.error(f"🚨 Detectados {len(clones)} pares de sensores redundantes!")
                    st.dataframe(pd.DataFrame(clones, columns=["Sensor A", "Sensor B", "Correlação"]).sort_values(by="Correlação", ascending=False))
                    st.session_state['lixo_para_remover'] = list(set([c[1] for c in clones]))
                else:
                    st.success("✅ Base saudável! Nenhum sensor clonado detectado.")
                    
        if 'lixo_para_remover' in st.session_state and st.session_state['lixo_para_remover']:
            st.warning("⚠️ Recomendamos remover os clones (Sensor B) para evitar matriz singular.")
            if st.checkbox("Confirmo a remoção dos clones."):
                if st.button("🗑️ Apagar Sensores"):
                    st.session_state['df_pi'] = st.session_state['df_pi'].drop(columns=st.session_state['lixo_para_remover'], errors='ignore')
                    st.session_state['msg_sucesso'] = "✅ Sensores redundantes apagados!"
                    st.session_state['lixo_para_remover'] = [] 
                    st.rerun() 
                    
        if 'msg_sucesso' in st.session_state:
            st.success(st.session_state['msg_sucesso'])

# ==========================================
# 2. LIMPEZA HEURÍSTICA
# ==========================================
elif menu == "🧹 2. Limpeza Heurística":
    st.title("🧹 Filtros de Engenharia")
    if st.session_state['df_pi'] is None:
        st.warning("⚠️ Carregue a base no Módulo 1.")
    else:
        st.subheader("A. Limites Específicos por Variável (Opcional)")
        with st.expander("➕ Adicionar Regra", expanded=False):
            termo = st.text_input("1. Buscar variável (ex: TG11):")
            if termo:
                cols_encontradas = [c for c in st.session_state['df_pi'].columns if termo.lower() in str(c).lower()]
                tags_sel = st.multiselect("2. Selecione as Tags:", cols_encontradas)
                colA, colB = st.columns(2)
                v_min = colA.number_input("Mínimo", value=None)
                v_max = colB.number_input("Máximo", value=None)
                if st.button("💾 Salvar Regra") and tags_sel:
                    st.session_state['limites_customizados'].append({'tags': tags_sel, 'minimo': v_min, 'maximo': v_max})
                    st.success("Regra adicionada!")

        if st.session_state['limites_customizados']:
            st.table(pd.DataFrame(st.session_state['limites_customizados']))
            if st.button("Limpar Regras"): st.session_state['limites_customizados'] = []; st.rerun()

        st.divider()
        st.subheader("B. Limpeza Global Automática")
        col1, col2 = st.columns(2)
        with col1:
            z = st.checkbox("Remover Zeros", value=True)
            n = st.checkbox("Remover Negativos", value=True)
        with col2:
            metodo = st.selectbox("Filtro de Outliers", ["Nenhum", "IQR", "Z-Score"])
            fator = st.number_input("Fator Sensibilidade", value=1.5 if metodo == "IQR" else 3.0)

        if st.button("🚀 Aplicar Limpeza Completa"):
            with st.spinner("Limpando histórico temporal e calculando estatísticas..."):
                df_bruto = st.session_state['df_pi']
                df_limpo = limpar_dados(df_bruto, z, n, metodo, fator, st.session_state['limites_customizados'])
                st.session_state['df_limpo'] = df_limpo
                st.success("✅ Dados limpos e prontos!")
                
                # --- A TABELA DE COMPARAÇÃO (ANTES vs DEPOIS) ---
                st.write("### 📊 Impacto dos Filtros na Base")
                
                # Força numérico no original para evitar erros no cálculo
                df_antes_num = df_bruto.apply(pd.to_numeric, errors='coerce')
                colunas_stats = ['count', 'mean', '50%', 'std', 'min', 'max']
                
                try:
                    desc_antes = df_antes_num.describe().T[colunas_stats].rename(columns={'50%': 'mediana'})
                    desc_depois = df_limpo.describe().T[colunas_stats].rename(columns={'50%': 'mediana'})
                    
                    # Concatena as tabelas lado a lado com os cabeçalhos coloridos
                    comparativo = pd.concat([desc_antes, desc_depois], axis=1, keys=['🔴 ANTES', '🟢 DEPOIS']).round(2)
                    
                    # Mostra a tabela interativa na tela
                    st.dataframe(comparativo, use_container_width=True)
                except Exception as e:
                    st.warning("Aviso: Não foi possível gerar a tabela estatística de comparação com estes dados.")

                # --- O GRÁFICO DE AMOSTRA ---
                st.write("### 📈 Amostra Visual (Primeira Tag da Lista)")
                fig, ax = plt.subplots(figsize=(10, 3))
                ptag = df_limpo.columns[0]
                ax.plot(df_bruto.index, df_antes_num[ptag], alpha=0.3, label="Original", color='red')
                ax.plot(df_limpo.index, df_limpo[ptag], alpha=0.8, label="Limpo", color='blue')
                ax.legend()
                st.pyplot(fig)

# ==========================================
# 3. MAPEAMENTO DE INDICADORES
# ==========================================
elif menu == "📝 3. Mapeamento de Indicadores":
    st.title("📝 Configuração de Sensores Virtuais")
    if st.session_state['df_pi'] is None:
        st.warning("⚠️ Carregue a base no Módulo 1.")
    else:
        with st.form("form_indicador"):
            nome_ind = st.text_input("Nome do Indicador (ex: Eficiência Térmica)")
            st.info("💡 A fórmula deve usar nomes genéricos. Ex: `(POTENCIA / COMBUSTIVEL) * 100`")
            formula = st.text_input("Fórmula Matemática Real", placeholder="Ex: (VarA / VarB) * 100")
            
            st.write("Selecione as Tags Reais do PI AF (Siga a ordem das variáveis na fórmula):")
            tags_selecionadas = st.multiselect("Tags Físicas", st.session_state['df_pi'].columns)

            if st.form_submit_button("💾 Salvar Indicador") and len(tags_selecionadas) > 0 and formula:
                st.session_state['mapeamento'].append({"nome": nome_ind, "formula": formula, "tags": tags_selecionadas})
                st.success(f"Indicador '{nome_ind}' salvo!")

        if st.session_state['mapeamento']:
            st.table(pd.DataFrame(st.session_state['mapeamento']))
            if st.button("Limpar Todos"): st.session_state['mapeamento'] = []; st.rerun()

# ==========================================
# 4. DASHBOARD DE PROCESSO
# ==========================================
elif menu == "📊 4. Dashboard de Processo":
    st.title("📊 Monitoramento de Performance")
    if not st.session_state['mapeamento']:
        st.info("ℹ️ Nenhum indicador mapeado. Vá ao Módulo 3.")
    else:
        df_target = st.session_state['df_limpo'] if st.session_state['df_limpo'] is not None else st.session_state['df_pi']
        
        # Filtro Global de Datas
        try:
            d_min, d_max = df_target.index.min().date(), df_target.index.max().date()
            datas = st.date_input("Filtrar Período de Análise", [d_min, d_max], min_value=d_min, max_value=d_max)
            if len(datas) == 2:
                df_target = df_target.loc[str(datas[0]):str(datas[1])]
        except Exception:
            pass # Se o índice não for data, ignora o filtro
        
        for item in st.session_state['mapeamento']:
            with st.expander(f"📈 Analisar: {item['nome']} (Fórmula: {item['formula']})", expanded=True):
                serie_calc = avaliar_formula_complexa(df_target, item['formula'], item['tags'])
                
                if serie_calc is not None and not serie_calc.isna().all():
                    serie_calc = serie_calc.dropna()
                    media, desvio = serie_calc.mean(), serie_calc.std()
                    
                    colA, colB, colC = st.columns(3)
                    colA.metric("Média", f"{media:.2f}")
                    colB.metric("Máximo", f"{serie_calc.max():.2f}")
                    colC.metric("Mínimo", f"{serie_calc.min():.2f}")

                    # Layout Colab: Carta Univariada + Histograma Lado a Lado
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [3, 1]})
                    
                    # Gráfico de Controle
                    ax1.plot(serie_calc.index, serie_calc.values, color='#1f77b4', alpha=0.8)
                    ax1.axhline(media, color='green', label=f'Média ({media:.2f})')
                    if pd.notna(desvio) and desvio > 0:
                        ax1.axhline(media + 3 * desvio, color='red', linestyle='--', label='LSC (+3σ)')
                        ax1.axhline(media - 3 * desvio, color='red', linestyle='--', label='LIC (-3σ)')
                    ax1.set_title(f"Carta de Controle - {item['nome']}")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Histograma
                    ax2.hist(serie_calc.values, bins=20, orientation='horizontal', edgecolor='white', color='orange', alpha=0.8)
                    ax2.axhline(media, color='green', linewidth=2)
                    if pd.notna(desvio) and desvio > 0:
                        ax2.axhline(media + 3 * desvio, color='red', linestyle='--')
                        ax2.axhline(media - 3 * desvio, color='red', linestyle='--')
                    ax2.set_title("Distribuição")
                    ax2.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                else:
                    st.error("Erro ao calcular a fórmula. Verifique as tags ou se houve divisão por zero.")

# ==========================================
# 5. ANÁLISE AVANÇADA (PCA E HOTELLING)
# ==========================================
elif menu == "🧪 5. Análise Avançada (PCA/T²)":
    st.title("🧪 Diagnóstico Avançado")
    if st.session_state['df_limpo'] is None:
        st.warning("⚠️ Execute o Módulo 2 primeiro para evitar erros matemáticos com dados sujos.")
    else:
        df_target = st.session_state['df_limpo']
        tab1, tab2 = st.tabs(["📉 PCA (Componentes)", "🎯 Carta T² (Hotelling)"])

        with tab1:
            termo_pca = st.text_input("Filtrar tags para o PCA (ex: TG11):", key="pca_input")
            if st.button("🚀 Processar PCA") and termo_pca:
                cols = [c for c in df_target.columns if termo_pca.lower() in str(c).lower()]
                if len(cols) > 1:
                    with st.spinner("Processando Matrizes..."):
                        df_pca = df_target[cols].dropna()
                        df_pca = df_pca.loc[:, df_pca.std() > 1e-6]
                        
                        pca = PCA()
                        pca_data = pca.fit_transform(StandardScaler().fit_transform(df_pca))
                        var_exp = pca.explained_variance_ratio_ * 100

                        fig, ax = plt.subplots(figsize=(8, 4))
                        scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=range(len(pca_data)), cmap='viridis', alpha=0.5)
                        ax.set_title("Mapa de Estado Operacional (PC1 vs PC2)")
                        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
                        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
                        plt.colorbar(scatter, label='Evolução Temporal')
                        st.pyplot(fig)
                        
                        st.write("### ⚖️ Top 5 Sensores que mais influenciam (Loadings)")
                        loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(var_exp))], index=df_pca.columns)
                        colA, colB = st.columns(2)
                        with colA:
                            st.write("**Impacto no Eixo PC1:**")
                            st.dataframe(loadings['PC1'].abs().sort_values(ascending=False).head(5))
                        with colB:
                            st.write("**Impacto no Eixo PC2:**")
                            st.dataframe(loadings['PC2'].abs().sort_values(ascending=False).head(5))

        with tab2:
            termo_t2 = st.text_input("Agrupar equipamentos para o T² (ex: Turbina):", key="t2_input")
            confianca = st.slider("Confiança Estatística", 0.90, 0.999, 0.99)
            if st.button("🚀 Calcular Carta T²") and termo_t2:
                cols = [c for c in df_target.columns if termo_t2.lower() in str(c).lower()]
                if len(cols) > 1:
                    with st.spinner("Calculando Covariância..."):
                        try:
                            df_t2 = df_target[cols].dropna()
                            df_t2 = df_t2.loc[:, df_t2.std() > 1e-6]
                            diff = df_t2 - df_t2.mean()
                            inv_cov = np.linalg.inv(df_t2.cov())
                            t2_valores = np.sum((np.dot(diff, inv_cov)) * diff, axis=1)
                            lsc_t2 = chi2.ppf(confianca, df=df_t2.shape[1])

                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(df_t2.index, t2_valores, color='purple')
                            ax.axhline(lsc_t2, color='red', linewidth=2, label=f'Limite T² ({confianca*100:.1f}%)')
                            ax.set_title(f"Carta T² de Hotelling - {len(cols)} Sensores")
                            ax.legend()
                            st.pyplot(fig)
                        except np.linalg.LinAlgError:
                            st.error("❌ ERRO: Matriz singular. Há sensores perfeitamente clonados neste grupo. Volte ao Módulo 1 para removê-los.")
