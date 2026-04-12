import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

# ==========================================
# 0. CONFIGURAÇÃO E ESTADO
# ==========================================
st.set_page_config(page_title="OPERA - Process Control", layout="wide", page_icon="🏭")

# Inicialização do session_state
defaults = {
    'df_pi': None,
    'df_limpo': None,
    'mapeamento': {},          # dict: nome -> {formula, tags}
    'limites_customizados': [],
    'limites_texto': "",       # string no formato legado (memória persistente)
    'lixo_para_remover': [],
    'msg_sucesso': None,
    'pca_n_top': 10,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# FUNÇÕES DE BACKEND
# ==========================================

# ---- Processador da Base PI AF (Célula 2) ----
def preparar_base_pi(arquivo_upload):
    """Lê o Excel bruto exportado do PI AF, extrai e limpa os nomes das tags,
    estrutura o DataFrame com índice temporal e remove tags 100% mortas."""
    try:
        df_raw = pd.read_excel(arquivo_upload, header=0)
    except Exception as e:
        st.error(f"❌ Erro ao ler o Excel: {e}")
        return None

    nomes_brutos = df_raw.iloc[:, 0].astype(str)
    dados = df_raw.iloc[:, 1:].copy()

    def extrair_partes(caminho):
        return [p.strip() for p in caminho.replace("\\\\", "").replace("\\", "/").replace("|", "/").split("/") if p.strip()]

    rotulos_finais = [extrair_partes(n)[-1] if extrair_partes(n) else n for n in nomes_brutos]
    duplicados = {k for k, v in Counter(rotulos_finais).items() if v > 1}

    nomes_unicos = []
    for nome in nomes_brutos:
        partes = extrair_partes(nome)
        rotulo = partes[-1] if partes else nome
        if rotulo in duplicados and len(partes) >= 2:
            nomes_unicos.append(f"{rotulo}_{re.sub(r'[^A-Za-z0-9_]', '', partes[-2])}")
        else:
            nomes_unicos.append(rotulo)

    ainda_dup = {k for k, v in Counter(nomes_unicos).items() if v > 1}
    if ainda_dup:
        ocorrencias = Counter()
        novos_nomes = []
        for n in nomes_unicos:
            if n in ainda_dup:
                ocorrencias[n] += 1
                novos_nomes.append(f"{n}_{ocorrencias[n]}")
            else:
                novos_nomes.append(n)
        nomes_unicos = novos_nomes

    datas = pd.to_datetime(dados.columns, errors="coerce")
    dados_numericos = dados.apply(pd.to_numeric, errors="coerce").values.T
    idx_datas = [d.normalize() if pd.notna(d) else pd.NaT for d in datas]

    df = pd.DataFrame(data=dados_numericos, index=idx_datas, columns=nomes_unicos)
    df = df[df.index.notna()].sort_index()
    df = df.loc[:, ~((df == 0) | df.isna()).all(axis=0)]
    return df


# ---- Motor Heurístico (Célula 3) ----
def parsear_limites_por_variavel(texto_limites):
    """Converte string de limites no formato 'TAG1, TAG2 [min,max]' para dict."""
    resultado = {}
    texto = str(texto_limites).strip()
    if not texto:
        return resultado
    for bloco in texto.split(']'):
        if not bloco.strip() or '[' not in bloco:
            continue
        nomes_str, limites_str = bloco.split('[', 1)
        nomes_str = nomes_str.replace(';', ',')
        lista_nomes = [n.strip().upper() for n in nomes_str.split(',') if n.strip()]
        limites = limites_str.split(',')
        if len(limites) >= 2:
            try:
                inf = float(limites[0].strip()) if limites[0].strip() else None
            except Exception:
                inf = None
            try:
                sup = float(limites[1].strip()) if limites[1].strip() else None
            except Exception:
                sup = None
            for nome in lista_nomes:
                resultado[nome] = (inf, sup)
    return resultado


def encontrar_limites_para_tag(tag, limites_cfg):
    tag_upper = str(tag).upper()
    if tag_upper in limites_cfg:
        return limites_cfg[tag_upper]
    for chave, lims in limites_cfg.items():
        if chave in tag_upper:
            return lims
    return (None, None)


def limpar_serie(serie, remover_zeros, remover_negativos, lim_inf, lim_sup,
                 usar_outlier, metodo_outlier, fator_iqr, limite_zscore,
                 preencher_ultimo, usar_rolling, janela_rolling, pct_minimo):
    s = pd.to_numeric(serie.copy(), errors="coerce")
    if remover_zeros:
        s = s.mask(np.isclose(s, 0, atol=1e-5))
    if remover_negativos:
        s = s.mask(s < 0)
    if lim_inf is not None:
        s = s.mask(s < lim_inf)
    if lim_sup is not None:
        s = s.mask(s > lim_sup)
    s_valid = s.dropna()
    if len(s_valid) > 3 and usar_outlier:
        if metodo_outlier == "IQR":
            q1, q3 = s_valid.quantile(0.25), s_valid.quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                s = s.mask((s < q1 - fator_iqr * iqr) | (s > q3 + fator_iqr * iqr))
        elif metodo_outlier == "Z-Score":
            media, desvio = s_valid.mean(), s_valid.std()
            if pd.notna(desvio) and desvio > 0:
                s = s.mask(((s - media) / desvio).abs() > limite_zscore)
    if preencher_ultimo:
        s = s.ffill()
    if usar_rolling and janela_rolling > 1:
        s = s.rolling(window=janela_rolling, min_periods=1).mean()
    return s if s.notna().mean() >= pct_minimo else pd.Series(index=s.index, dtype=float)


def gerar_dataframe_limpo(df_original, tags_validas, cfg):
    """Aplica o motor heurístico a todas as tags e retorna DataFrame limpo."""
    limites_cfg = parsear_limites_por_variavel(st.session_state.get('limites_texto', ''))
    # Mescla com limites da UI (formato lista de dicts)
    for regra in st.session_state.get('limites_customizados', []):
        for tag in regra['tags']:
            limites_cfg[tag.upper()] = (regra.get('minimo'), regra.get('maximo'))

    df_saida = pd.DataFrame(index=df_original.index)
    for tag in tags_validas:
        s_orig = pd.to_numeric(df_original[tag], errors="coerce")
        lim_inf, lim_sup = encontrar_limites_para_tag(tag, limites_cfg)
        if cfg['usar_limpeza'] or lim_inf is not None or lim_sup is not None:
            df_saida[tag] = limpar_serie(
                s_orig, cfg['remover_zeros'], cfg['remover_negativos'],
                lim_inf, lim_sup,
                cfg['usar_outlier'], cfg['metodo_outlier'],
                cfg['fator_iqr'], cfg['limite_zscore'],
                cfg['preencher_ultimo'], cfg['usar_rolling'],
                cfg['janela_rolling'], cfg['pct_minimo']
            )
        else:
            df_saida[tag] = s_orig
    return df_saida.dropna(how="all")


# ---- Motor Calculador de Indicadores (Célula 5) ----
def calcular_indicador(df_plot, formula, tags_validas):
    """Tenta calcular a série temporal do indicador via eval da fórmula.
    Retorna dict {nome_unidade: serie} ou None se falhar."""
    parte_direita = formula.split("=", 1)[1] if "=" in formula else formula
    parte_direita = parte_direita.strip()
    termos_ignorar = {'SOMA', 'MEDIA', 'SE', 'MAX', 'MIN', 'IF', 'AND', 'OR'}
    palavras_brutas = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*%?', parte_direita)
    vars_esperadas = list(dict.fromkeys([p for p in palavras_brutas if p.upper() not in termos_ignorar]))
    if not vars_esperadas:
        vars_esperadas = ["VAR_UNICA"]

    resultados = {}
    if len(tags_validas) % len(vars_esperadas) == 0:
        chunk = len(tags_validas) // len(vars_esperadas)
        mapping = {var: tags_validas[i * chunk:(i + 1) * chunk] for i, var in enumerate(vars_esperadas)}

        for k in range(chunk):
            tag_principal = mapping[vars_esperadas[0]][k]
            maq = tag_principal.split('_')[-1] if '_' in tag_principal else f"Unid_{k + 1}"

            df_eval = pd.DataFrame()
            expressao = parte_direita
            vars_ordenadas = sorted(vars_esperadas, key=len, reverse=True)
            for var in vars_ordenadas:
                safe_var = re.sub(r'[^a-zA-Z0-9_]', '', var) or "VAR_SYMB"
                df_eval[safe_var] = df_plot[mapping[var][k]]
                expressao = re.sub(fr'\b{re.escape(var)}\b', safe_var, expressao)

            try:
                serie = df_eval.eval(expressao, engine='python')
            except Exception:
                local_dict = {col: df_eval[col] for col in df_eval.columns}
                try:
                    serie = eval(expressao, {"__builtins__": None, "np": np, "pd": pd}, local_dict)
                except Exception as e:
                    return None, str(e)

            serie = serie.replace([np.inf, -np.inf], np.nan)
            resultados[maq] = serie

        return resultados, None
    return None, "Número de tags incompatível com a fórmula."


# ---- PCA (Célula 6) ----
def executar_pca(df, cols):
    df_temp = df[cols].apply(pd.to_numeric, errors='coerce')
    df_temp = df_temp.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    cols_validas = df_temp.columns[df_temp.std() > 1e-6]
    df_final = df_temp[cols_validas]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_final)
    pca = PCA()
    pca_data = pca.fit_transform(X_scaled)
    return pca, pca_data, df_final


# ---- T² de Hotelling (Célula 7) ----
def calcular_t2(df, cols, nivel_confianca):
    df_clean = df[cols].apply(pd.to_numeric, errors='coerce')
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    df_clean = df_clean.loc[:, df_clean.std() > 1e-6]
    if df_clean.shape[1] < 2:
        return None, None, None, "Menos de 2 variáveis após limpeza."
    media_vec = df_clean.mean()
    cov_matriz = df_clean.cov()
    try:
        inv_cov = np.linalg.inv(cov_matriz)
    except np.linalg.LinAlgError:
        return None, None, None, "Matriz de covariância singular."
    diff = df_clean - media_vec
    t2_valores = np.sum((np.dot(diff, inv_cov)) * diff, axis=1)
    lsc_t2 = chi2.ppf(nivel_confianca, df=df_clean.shape[1])
    var_principal = df_clean.var().idxmax()
    return t2_valores, lsc_t2, df_clean, var_principal


# ==========================================
# MENU LATERAL
# ==========================================
st.sidebar.title("🏭 OPERA v2.0")
menu = st.sidebar.selectbox("Navegação:", [
    "📂 1. Carga e Auditoria",
    "🧹 2. Limpeza Heurística",
    "📝 3. Mapeamento de Indicadores",
    "📊 4. Dashboard CEP",
    "🧪 5. Análise Avançada (PCA / T²)",
])
st.sidebar.divider()
st.sidebar.info("Monitoramento estatístico multivariado de processo.")

# ==========================================
# 1. CARGA E AUDITORIA
# ==========================================
if menu == "📂 1. Carga e Auditoria":
    st.title("📂 Carga e Auditoria de Sensores")

    tipo_arquivo = st.radio("Formato do arquivo:", ["PKL (base já processada)", "XLSX (exportação bruta do PI AF)"], horizontal=True)

    arquivo = st.file_uploader(
        "Arraste seu arquivo",
        type=["pkl", "xlsx"],
        help="PKL = base já limpa pelo processador. XLSX = exportação direta do PI AF (com nomes de tags nas linhas)."
    )

    if arquivo:
        with st.spinner("Processando base de dados..."):
            try:
                if arquivo.name.endswith('.pkl') or tipo_arquivo.startswith("PKL"):
                    df = pd.read_pickle(arquivo)
                else:
                    df = preparar_base_pi(arquivo)

                if df is not None:
                    if not pd.api.types.is_datetime64_any_dtype(df.index):
                        try:
                            df.index = pd.to_datetime(df.index)
                        except Exception:
                            pass
                    st.session_state['df_pi'] = df
                    st.success(f"✅ Base carregada: **{df.shape[0]}** instantes × **{df.shape[1]}** tags.")
                    st.dataframe(df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Erro ao carregar arquivo: {e}")

    st.divider()

    # ---- Auditoria de Clones Exatos (Célula de Auditoria) ----
    st.subheader("🕵️ Auditoria de Redundância — Sensores Clonados")
    if st.session_state['df_pi'] is None:
        st.info("Carregue a base acima para habilitar a auditoria.")
    else:
        col_audit1, col_audit2 = st.columns(2)
        with col_audit1:
            limiar = st.number_input("Limiar de correlação para redundância", min_value=0.90, max_value=1.0,
                                     value=0.99, step=0.01,
                                     help="Pares com correlação acima deste valor serão sinalizados.")
        with col_audit2:
            st.write("")
            st.write("")
            executar = st.button("🔍 Executar Auditoria Completa")

        if executar:
            df_audit = st.session_state['df_pi']

            # 1. Nomes duplicados
            st.markdown("**1️⃣ Nomes de Tags Duplicados**")
            mascara_nomes = df_audit.columns.duplicated(keep='first')
            nomes_dup = df_audit.columns[mascara_nomes].unique()
            if len(nomes_dup) > 0:
                st.warning(f"⚠️ {sum(mascara_nomes)} cabeçalhos repetidos: {list(nomes_dup[:10])}")
            else:
                st.success("✅ Nenhum nome de tag repetido.")

            # 2. Dados clonados (100% idênticos)
            st.markdown("**2️⃣ Dados Clonados (Histórico 100% Idêntico)**")
            try:
                mascara_dados = df_audit.T.duplicated(keep='first')
                tags_clonadas = df_audit.columns[mascara_dados].tolist()
                if tags_clonadas:
                    st.warning(f"🚨 {len(tags_clonadas)} variáveis são cópias exatas.")
                    st.write(tags_clonadas[:10])
                else:
                    st.success("✅ Nenhum sensor com dados perfeitamente clonados.")
            except MemoryError:
                st.warning("⚠️ Base muito pesada para cruzamento completo na memória.")

            # 3. Correlação alta (configurável)
            st.markdown(f"**3️⃣ Pares com Correlação ≥ {limiar}**")
            with st.spinner("Calculando matriz de correlação..."):
                df_num = df_audit.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
                df_num = df_num.ffill().bfill()
                df_num = df_num.loc[:, df_num.std() > 1e-6]

                if not df_num.empty:
                    matriz_corr = df_num.corr().abs()
                    upper = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))
                    pares = []
                    for coluna in upper.columns:
                        matches = upper[coluna][upper[coluna] >= limiar]
                        for linha, valor in matches.items():
                            pares.append((linha, coluna, round(valor, 4)))

                    if pares:
                        pares.sort(key=lambda x: x[2], reverse=True)
                        df_pares = pd.DataFrame(pares, columns=["Sensor A (Manter)", "Sensor B (Remover)", "Correlação"])
                        st.error(f"🚨 {len(pares)} pares redundantes detectados!")
                        st.dataframe(df_pares, use_container_width=True)
                        st.session_state['lixo_para_remover'] = list(set([p[1] for p in pares]))
                    else:
                        st.success(f"✅ Nenhum par ultrapassou {limiar*100:.0f}% de correlação.")

        # Botão de limpeza
        if st.session_state.get('lixo_para_remover'):
            st.divider()
            st.warning(f"⚠️ {len(st.session_state['lixo_para_remover'])} sensores marcados para remoção.")
            if st.checkbox("✅ Confirmo a remoção dos clones."):
                if st.button("🗑️ Remover Sensores Redundantes"):
                    st.session_state['df_pi'] = st.session_state['df_pi'].drop(
                        columns=st.session_state['lixo_para_remover'], errors='ignore')
                    st.session_state['lixo_para_remover'] = []
                    st.success("✅ Sensores redundantes removidos! Base atualizada.")
                    st.rerun()

    if st.session_state.get('msg_sucesso'):
        st.success(st.session_state['msg_sucesso'])
        st.session_state['msg_sucesso'] = None


# ==========================================
# 2. LIMPEZA HEURÍSTICA
# ==========================================
elif menu == "🧹 2. Limpeza Heurística":
    st.title("🧹 Motor de Limpeza Heurística")

    if st.session_state['df_pi'] is None:
        st.warning("⚠️ Carregue a base no Módulo 1.")
        st.stop()

    df_base = st.session_state['df_pi']

    # ---- A. Limites específicos por variável ----
    st.subheader("A. Limites Específicos por Variável (Memória de Limites)")

    with st.expander("➕ Adicionar / Gerenciar Regras de Limite", expanded=False):
        termo_lim = st.text_input("Buscar variável (ex: pressao, TG11):", key="busca_limites")
        if termo_lim:
            cols_enc = [c for c in df_base.columns if termo_lim.lower() in str(c).lower()]
            if cols_enc:
                tags_sel = st.multiselect("Tags encontradas:", cols_enc, key="tags_lim_sel")
                cA, cB = st.columns(2)
                v_min = cA.number_input("Mínimo (deixe 0 p/ ignorar)", value=0.0, key="vmin_lim")
                v_max = cB.number_input("Máximo (deixe 0 p/ ignorar)", value=0.0, key="vmax_lim")
                usar_min = cA.checkbox("Aplicar Mínimo", value=False)
                usar_max = cB.checkbox("Aplicar Máximo", value=False)
                if st.button("💾 Salvar Regra") and tags_sel:
                    st.session_state['limites_customizados'].append({
                        'tags': tags_sel,
                        'minimo': v_min if usar_min else None,
                        'maximo': v_max if usar_max else None
                    })
                    st.success("✅ Regra adicionada!")
            else:
                st.info("Nenhuma tag encontrada com este termo.")

    if st.session_state['limites_customizados']:
        st.markdown("**Regras ativas:**")
        df_regras = pd.DataFrame(st.session_state['limites_customizados'])
        st.dataframe(df_regras, use_container_width=True)
        if st.button("🗑️ Limpar Todas as Regras"):
            st.session_state['limites_customizados'] = []
            st.rerun()

    st.divider()

    # ---- B. Configuração Global ----
    st.subheader("B. Configuração da Limpeza Global")

    col1, col2, col3 = st.columns(3)
    with col1:
        usar_limpeza = st.checkbox("Ativar Heurística Global", value=True)
        remover_zeros = st.checkbox("Remover Zeros", value=True)
        remover_negativos = st.checkbox("Remover Negativos", value=True)
    with col2:
        usar_outlier = st.checkbox("Filtro de Outliers", value=False)
        metodo_outlier = st.selectbox("Método Outlier", ["IQR", "Z-Score"])
        fator_iqr = st.number_input("Fator IQR", value=1.5, min_value=0.1)
        limite_zscore = st.number_input("Limite Z-Score", value=3.0, min_value=0.1)
    with col3:
        preencher_ultimo = st.checkbox("Forward Fill (último valor)", value=False)
        usar_rolling = st.checkbox("Suavização Rolling Mean", value=False)
        janela_rolling = st.number_input("Janela Rolling", value=5, min_value=2, step=1)
        pct_minimo = st.slider("% Mínimo de Dados Válidos", 0.0, 1.0, 0.0, 0.01)

    cfg = dict(
        usar_limpeza=usar_limpeza, remover_zeros=remover_zeros, remover_negativos=remover_negativos,
        usar_outlier=usar_outlier, metodo_outlier=metodo_outlier, fator_iqr=fator_iqr,
        limite_zscore=limite_zscore, preencher_ultimo=preencher_ultimo, usar_rolling=usar_rolling,
        janela_rolling=int(janela_rolling), pct_minimo=pct_minimo
    )

    if st.button("🚀 Aplicar Limpeza Completa"):
        with st.spinner("Aplicando filtros de engenharia..."):
            tags_validas = list(df_base.columns)
            df_limpo = gerar_dataframe_limpo(df_base, tags_validas, cfg)
            st.session_state['df_limpo'] = df_limpo
            st.session_state['cfg_limpeza'] = cfg
            st.success(f"✅ Limpeza concluída! {df_limpo.shape[1]} tags, {df_limpo.shape[0]} instantes.")

            # Tabela ANTES vs DEPOIS
            st.write("### 📊 Impacto dos Filtros (ANTES vs DEPOIS)")
            df_antes_num = df_base.apply(pd.to_numeric, errors='coerce')
            colunas_stats = ['count', 'mean', '50%', 'std', 'min', 'max']
            try:
                desc_antes = df_antes_num.describe().T[colunas_stats].rename(columns={'50%': 'mediana'})
                desc_depois = df_limpo.describe().T[colunas_stats].rename(columns={'50%': 'mediana'})
                comparativo = pd.concat([desc_antes, desc_depois], axis=1, keys=['🔴 ANTES', '🟢 DEPOIS']).round(2)
                st.dataframe(comparativo, use_container_width=True)
            except Exception:
                st.warning("Não foi possível gerar tabela comparativa.")

            # Amostra Visual
            st.write("### 📈 Amostra Visual (Primeira Tag)")
            ptag = df_limpo.columns[0]
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df_base.index, df_antes_num[ptag], alpha=0.3, label="Original", color='red')
            ax.plot(df_limpo.index, df_limpo[ptag], alpha=0.8, label="Limpo", color='blue')
            ax.set_title(ptag)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)


# ==========================================
# 3. MAPEAMENTO DE INDICADORES
# ==========================================
elif menu == "📝 3. Mapeamento de Indicadores":
    st.title("📝 Mapeamento de Indicadores de Processo")

    if st.session_state['df_pi'] is None:
        st.warning("⚠️ Carregue a base no Módulo 1.")
        st.stop()

    df_base = st.session_state['df_pi']

    # Upload opcional de lista de indicadores (lista_indicadores.xlsx)
    with st.expander("📥 Importar Lista de Indicadores (Excel Opcional)", expanded=False):
        arquivo_ind = st.file_uploader(
            "Faça upload do arquivo lista_indicadores.xlsx",
            type=["xlsx"],
            key="uploader_indicadores",
            help="Planilha com indicadores nas linhas 10+, nome na col E e fórmula nas colunas F:K."
        )
        nome_aba = st.text_input("Nome da aba", value="Indicadores")

        if arquivo_ind:
            try:
                df_ind = pd.read_excel(arquivo_ind, sheet_name=nome_aba, header=None)
                lista_importada = []
                for idx in range(9, min(100, len(df_ind))):
                    nome = str(df_ind.iloc[idx, 4]).strip()
                    if nome and "BENCHMARK" not in nome.upper() and nome != "nan":
                        formula_bruta = " ".join([
                            c for c in df_ind.iloc[idx, 5:11].dropna().astype(str).tolist()
                            if c.lower() != 'nan'
                        ])
                        lista_importada.append({"nome": nome, "formula_bruta": formula_bruta})

                st.success(f"✅ {len(lista_importada)} indicadores importados da planilha.")
                if st.button("Pré-carregar indicadores importados"):
                    for item in lista_importada:
                        if item['nome'] not in st.session_state['mapeamento']:
                            st.session_state['mapeamento'][item['nome']] = {
                                'formula': item['formula_bruta'],
                                'tags': []
                            }
                    st.success("Indicadores carregados! Complete as tags abaixo.")
                    st.rerun()
            except Exception as e:
                st.error(f"Erro ao ler a planilha: {e}")

    st.divider()

    # ---- Formulário de Criação / Edição ----
    st.subheader("Adicionar ou Editar Indicador")

    nomes_existentes = list(st.session_state['mapeamento'].keys())
    modo = st.radio("Modo:", ["Novo Indicador", "Editar Existente"], horizontal=True)

    if modo == "Editar Existente" and nomes_existentes:
        nome_edit = st.selectbox("Selecione o indicador:", nomes_existentes)
        item_edit = st.session_state['mapeamento'][nome_edit]
        nome_ind = nome_edit
        formula_default = item_edit['formula']
        tags_default = item_edit['tags']
    else:
        nome_ind = st.text_input("Nome do Indicador (ex: Eficiência Térmica TG11)")
        formula_default = ""
        tags_default = []

    formula = st.text_input("Fórmula Matemática", value=formula_default,
                            placeholder="Ex: (POTENCIA / COMBUSTIVEL) * 100")

    st.info("💡 **Busque e selecione as tags na ordem das variáveis da fórmula.**")
    termo_tag = st.text_input("Buscar tags (ex: Potencia, TG11):", key="busca_tags_mapeamento")
    if termo_tag:
        cols_enc = [c for c in df_base.columns
                    if termo_tag.lower() in str(c).lower()
                    and ((df_base[c].isna().sum() + (df_base[c] == 0).sum()) / len(df_base)) < 0.99]
        tags_sel = st.multiselect("Tags encontradas (selecione na ordem da fórmula):", cols_enc,
                                  default=[t for t in tags_default if t in cols_enc],
                                  key="tags_mapeamento_sel")
    else:
        tags_sel = st.multiselect("Tags Físicas (busque acima):", df_base.columns,
                                  default=tags_default, key="tags_mapeamento_all")

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        salvar = st.button("💾 Salvar Indicador", type="primary")
    with col_btn2:
        if modo == "Editar Existente" and nomes_existentes:
            if st.button("🗑️ Remover este Indicador"):
                del st.session_state['mapeamento'][nome_edit]
                st.success(f"Indicador '{nome_edit}' removido.")
                st.rerun()

    if salvar:
        if not nome_ind:
            st.error("Defina um nome para o indicador.")
        elif not formula:
            st.error("Defina a fórmula matemática.")
        elif not tags_sel:
            st.error("Selecione ao menos uma tag.")
        else:
            st.session_state['mapeamento'][nome_ind] = {'formula': formula, 'tags': tags_sel}
            st.success(f"✅ Indicador '{nome_ind}' salvo!")
            st.rerun()

    st.divider()

    # ---- Tabela de Indicadores Mapeados ----
    st.subheader("📋 Indicadores Mapeados")
    if st.session_state['mapeamento']:
        rows = []
        for nome, item in st.session_state['mapeamento'].items():
            status = "✅ Completo" if item['tags'] else "⏳ Pendente (sem tags)"
            rows.append({"Indicador": nome, "Fórmula": item['formula'],
                         "Tags": ", ".join(item['tags']), "Status": status})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if st.button("🗑️ Limpar Todos os Indicadores"):
            st.session_state['mapeamento'] = {}
            st.rerun()
    else:
        st.info("Nenhum indicador mapeado ainda.")


# ==========================================
# 4. DASHBOARD CEP
# ==========================================
elif menu == "📊 4. Dashboard CEP":
    st.title("📊 Dashboard CEP — Controle Estatístico de Processo")

    if not st.session_state['mapeamento']:
        st.info("ℹ️ Nenhum indicador mapeado. Vá ao Módulo 3.")
        st.stop()

    df_target = st.session_state['df_limpo'] if st.session_state['df_limpo'] is not None else st.session_state['df_pi']
    if df_target is None:
        st.warning("⚠️ Carregue a base no Módulo 1.")
        st.stop()

    # ---- Filtro de Datas ----
    st.subheader("⏱️ Janela de Tempo")
    try:
        d_min = df_target.index.min().date()
        d_max = df_target.index.max().date()
        datas = st.date_input("Período de análise:", [d_min, d_max], min_value=d_min, max_value=d_max)
        if len(datas) == 2:
            df_target = df_target.loc[str(datas[0]):str(datas[1])]
            st.caption(f"📌 Analisando **{len(df_target)}** amostras — de {datas[0]} a {datas[1]}")
    except Exception:
        st.caption("Índice não é datetime; filtro de data ignorado.")

    st.divider()

    # ---- Loop por indicador ----
    for nome_ind, item in st.session_state['mapeamento'].items():
        formula = item['formula']
        tags_validas = [t for t in item['tags'] if t in df_target.columns]

        with st.expander(f"📈 {nome_ind}  —  `{formula}`", expanded=True):
            if not tags_validas:
                st.warning("⚠️ Nenhuma tag válida encontrada na base atual.")
                continue

            resultados, erro = calcular_indicador(df_target, formula, tags_validas)

            if erro or not resultados:
                # Fallback: plota as tags individuais
                st.warning(f"⚠️ Não foi possível calcular a fórmula ({erro}). Exibindo tags brutas.")
                fig, axes = plt.subplots(1, 2, figsize=(14, 4))
                for tag in tags_validas:
                    serie_raw = df_target[tag].dropna()
                    axes[0].plot(serie_raw.index, serie_raw.values, alpha=0.7, label=tag[:20])
                axes[0].set_title(f"{nome_ind} — Variáveis Brutas")
                axes[0].legend(fontsize=8)
                axes[0].grid(True, alpha=0.3)
                serie_ref = df_target[tags_validas[0]].dropna()
                axes[1].hist(serie_ref.values, bins=20, orientation='horizontal',
                             edgecolor='white', color='orange', alpha=0.8)
                axes[1].set_title("Distribuição (Var Principal)")
                axes[1].grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
                continue

            n_maq = len(resultados)
            fig, axes = plt.subplots(nrows=n_maq, ncols=2, figsize=(15, 4 * n_maq),
                                     sharey='row', squeeze=False)

            metricas_globais = []
            for i, (maq, serie) in enumerate(resultados.items()):
                serie = serie.dropna()
                if serie.empty:
                    axes[i, 0].set_title(f"[{maq}] SEM DADOS VÁLIDOS")
                    continue

                media = serie.mean()
                desvio = serie.std() if pd.notna(serie.std()) else 0
                lsc = media + 3 * desvio
                lic = media - 3 * desvio
                metricas_globais.append({"Unidade": maq, "Média": round(media, 3),
                                         "Desvio": round(desvio, 3), "LSC": round(lsc, 3),
                                         "LIC": round(lic, 3), "N": len(serie)})

                # Carta de Controle
                axes[i, 0].plot(serie.index, serie.values, marker='o', markersize=2,
                                alpha=0.7, color='#1f77b4')
                axes[i, 0].axhline(media, color='green', label=f'Média: {media:.2f}')
                if desvio > 0:
                    axes[i, 0].axhline(lsc, linestyle='--', color='red', label='LSC (3σ)')
                    axes[i, 0].axhline(lic, linestyle='--', color='red', label='LIC (3σ)')
                    # Pontos fora de controle
                    fora = serie[(serie > lsc) | (serie < lic)]
                    axes[i, 0].scatter(fora.index, fora.values, color='red', s=20, zorder=5)
                axes[i, 0].set_title(f"[{maq}] Carta de Controle — {nome_ind}", fontweight='bold')
                axes[i, 0].legend(loc='upper right', fontsize=8)
                axes[i, 0].grid(True, alpha=0.2)

                # Histograma
                axes[i, 1].hist(serie.values, bins=15, orientation='horizontal',
                                edgecolor='white', alpha=0.8, color='orange')
                axes[i, 1].axhline(media, linewidth=2, color='green')
                if desvio > 0:
                    axes[i, 1].axhline(lsc, linestyle='--', color='red')
                    axes[i, 1].axhline(lic, linestyle='--', color='red')
                axes[i, 1].set_title(f"[{maq}] Distribuição", fontweight='bold')
                axes[i, 1].grid(True, alpha=0.2)

            plt.suptitle(f"{nome_ind}\nFórmula: {formula}  |  {len(df_target)} amostras",
                         fontsize=13, fontweight='bold', y=1.01)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            if metricas_globais:
                st.dataframe(pd.DataFrame(metricas_globais), use_container_width=True)


# ==========================================
# 5. ANÁLISE AVANÇADA (PCA / T²)
# ==========================================
elif menu == "🧪 5. Análise Avançada (PCA / T²)":
    st.title("🧪 Diagnóstico Avançado — PCA e T² de Hotelling")

    df_avancado = st.session_state['df_limpo'] if st.session_state['df_limpo'] is not None else st.session_state['df_pi']
    if df_avancado is None:
        st.warning("⚠️ Carregue a base no Módulo 1.")
        st.stop()
    if st.session_state['df_limpo'] is None:
        st.info("💡 Recomenda-se executar o Módulo 2 (Limpeza) antes para evitar erros com dados sujos.")

    tab1, tab2 = st.tabs(["📉 PCA — Componentes Principais", "🎯 T² de Hotelling"])

    # ---- PCA ----
    with tab1:
        st.subheader("Análise de Componentes Principais (PCA)")

        termo_pca = st.text_input("Filtrar tags para PCA (ex: TG11, Vazao):", key="pca_input")
        n_top = st.number_input("Top N sensores nos Loadings:", min_value=5, max_value=50, value=10, key="pca_top")

        if st.button("🚀 Processar PCA", key="btn_pca"):
            if not termo_pca:
                st.error("Digite um termo de busca.")
            else:
                cols_pca = [c for c in df_avancado.columns if termo_pca.lower() in str(c).lower()]
                if len(cols_pca) < 2:
                    st.error(f"Encontrado apenas {len(cols_pca)} tag(s) com '{termo_pca}'. PCA precisa de ao menos 2.")
                else:
                    st.info(f"✅ {len(cols_pca)} tags encontradas.")
                    with st.spinner("Calculando PCA..."):
                        pca, pca_data, df_pca = executar_pca(df_avancado, cols_pca)
                        var_exp = pca.explained_variance_ratio_ * 100
                        cum_var = np.cumsum(var_exp)

                        descartadas = len(cols_pca) - len(df_pca.columns)
                        if descartadas > 0:
                            st.warning(f"⚠️ {descartadas} tags descartadas por falta de variância (sensores estáticos).")

                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

                        # Scree Plot
                        n_show = min(20, len(var_exp))
                        ax1.bar(range(1, n_show + 1), var_exp[:n_show], alpha=0.5, label='Individual')
                        ax1.step(range(1, n_show + 1), cum_var[:n_show], where='mid',
                                 label='Acumulada', color='red')
                        ax1.set_title("Variância Explicada por Componente")
                        ax1.set_xlabel("PC")
                        ax1.set_ylabel("% Variância")
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)

                        # Scatter PC1 vs PC2
                        scatter = ax2.scatter(pca_data[:, 0], pca_data[:, 1],
                                              c=range(len(pca_data)), cmap='viridis', alpha=0.5)
                        ax2.set_title(f"Mapa Operacional (PC1 vs PC2)\nFiltro: {termo_pca}")
                        ax2.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
                        ax2.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
                        plt.colorbar(scatter, ax=ax2, label='Evolução Temporal')
                        ax2.grid(True, alpha=0.3)

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        # Loadings
                        loadings = pd.DataFrame(
                            pca.components_.T,
                            columns=[f'PC{i + 1}' for i in range(len(var_exp))],
                            index=df_pca.columns
                        )
                        colA, colB = st.columns(2)
                        with colA:
                            st.write(f"**⚖️ Top {int(n_top)} — PC1 ({var_exp[0]:.1f}%)**")
                            st.dataframe(loadings['PC1'].abs().sort_values(ascending=False).head(int(n_top)).round(4))
                        with colB:
                            st.write(f"**⚖️ Top {int(n_top)} — PC2 ({var_exp[1]:.1f}%)**")
                            st.dataframe(loadings['PC2'].abs().sort_values(ascending=False).head(int(n_top)).round(4))

    # ---- T² de Hotelling ----
    with tab2:
        st.subheader("Carta T² de Hotelling (Controle Multivariado)")

        termo_t2 = st.text_input("Agrupar sensores para T² (ex: Turbina, TG11):", key="t2_input")
        confianca = st.slider("Confiança Estatística", 0.90, 0.999, 0.99, 0.001, key="t2_conf")

        if st.button("🚀 Calcular Carta T²", key="btn_t2"):
            if not termo_t2:
                st.error("Digite um termo de busca.")
            else:
                cols_t2 = [c for c in df_avancado.columns if termo_t2.lower() in str(c).lower()]
                if len(cols_t2) < 2:
                    st.error("O T² exige ao menos 2 variáveis. Tente uma busca mais ampla.")
                else:
                    st.info(f"✅ {len(cols_t2)} tags encontradas.")
                    with st.spinner("Calculando Covariância..."):
                        t2_valores, lsc_t2, df_clean, var_principal = calcular_t2(df_avancado, cols_t2, confianca)

                        if t2_valores is None:
                            st.error(f"❌ {var_principal}")  # var_principal = msg de erro aqui
                        else:
                            p = df_clean.shape[1]
                            n = df_clean.shape[0]

                            # Carta Univariada (variável mais instável)
                            serie_uni = df_clean[var_principal]
                            media_uni = serie_uni.mean()
                            desvio_uni = serie_uni.std()
                            lsc_uni = media_uni + 3 * desvio_uni
                            lic_uni = media_uni - 3 * desvio_uni

                            tempo = df_clean.index if pd.api.types.is_datetime64_any_dtype(df_clean.index) \
                                else range(n)

                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

                            # Univariada
                            ax1.plot(tempo, serie_uni.values, color='#1f77b4', alpha=0.8, linewidth=1.2, label='Sinal Real')
                            ax1.axhline(media_uni, color='green', label=f'Média ({media_uni:.2f})')
                            ax1.axhline(lsc_uni, color='red', linestyle='--', label='LSC (+3σ)')
                            ax1.axhline(lic_uni, color='red', linestyle='--', label='LIC (-3σ)')
                            falhas_uni = serie_uni[(serie_uni > lsc_uni) | (serie_uni < lic_uni)]
                            ax1.scatter(falhas_uni.index, falhas_uni.values, color='red', s=20, zorder=5)
                            ax1.set_title(f"Carta Univariada: {var_principal}", fontweight='bold')
                            ax1.set_ylabel("Valor")
                            ax1.legend(loc='upper right', fontsize=9)
                            ax1.grid(True, alpha=0.3)

                            # T²
                            t2_serie = pd.Series(t2_valores, index=df_clean.index)
                            ax2.plot(tempo, t2_valores, color='purple', alpha=0.8, linewidth=1.2, label='Índice T²')
                            ax2.axhline(lsc_t2, color='red', linewidth=2,
                                        label=f'LSC Hotelling ({confianca * 100:.1f}%: {lsc_t2:.2f})')
                            falhas_t2 = t2_serie[t2_serie > lsc_t2]
                            ax2.scatter(falhas_t2.index, falhas_t2.values, color='red', s=20, zorder=5)
                            ax2.set_title(f"T² de Hotelling — {p} variáveis", fontweight='bold')
                            ax2.set_ylabel("T²")
                            ax2.set_xlabel("Evolução Temporal")
                            ax2.legend(loc='upper right', fontsize=9)
                            ax2.grid(True, alpha=0.3)

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                            # Relatório de Alarmes
                            pct_uni = len(falhas_uni) / n * 100
                            pct_t2 = len(falhas_t2) / n * 100

                            st.divider()
                            st.markdown("### 📊 Relatório de Alarmes: Univariado vs Multivariado")
                            cA, cB = st.columns(2)
                            cA.metric(f"Alarme Univariado ({var_principal[:20]}...)",
                                      f"{len(falhas_uni)} anomalias", f"{pct_uni:.2f}%")
                            cB.metric("Alarme Multivariado (T² Hotelling)",
                                      f"{len(falhas_t2)} anomalias", f"{pct_t2:.2f}%")

                            if pct_t2 > pct_uni:
                                st.warning("⚠️ **DIAGNÓSTICO:** O T² encontrou falhas **ocultas**! "
                                           "A relação física entre as variáveis quebrou sem que nenhuma isolada disparasse alarme.")
                            elif pct_t2 < pct_uni:
                                st.success("✅ **DIAGNÓSTICO:** O T² filtrou **falsos positivos**! "
                                           "Uma mudança de carga acionou o alarme univariado, mas a física do conjunto foi respeitada.")
                            else:
                                st.info("ℹ️ Alarmes univariado e multivariado estão alinhados.")
