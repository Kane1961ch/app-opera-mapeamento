import unicodedata
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


def normalizar(texto):
    """Remove acentos e converte para minusculas para busca tolerante."""
    return unicodedata.normalize("NFD", str(texto)).encode("ascii", "ignore").decode("ascii").lower()


def buscar_tags(colunas, termo):
    """Busca tags pelo termo, tolerante a acentos e maiusculas/minusculas."""
    termo_n = normalizar(termo)
    return [c for c in colunas if termo_n in normalizar(c)]

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
    '_lista_importada': [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# FUNÇÕES DE BACKEND
# ==========================================

# ---- Processador da Base PI AF ----
def preparar_base_pi(arquivo_upload):
    """
    Lê o Excel bruto do PI AF no formato:
      Linha 0: "data" | timestamp1 | timestamp2 | ...
      Linha 1+: caminho_completo_tag | valor1 | valor2 | ...

    Retorna DataFrame com índice DatetimeIndex e colunas = nomes limpos das tags.
    """
    try:
        if hasattr(arquivo_upload, "seek"):
            arquivo_upload.seek(0)
        df_raw = pd.read_excel(arquivo_upload, header=None)
    except Exception as e:
        st.error(f"❌ Erro ao ler o Excel: {e}")
        return None

    # Linha 0: col 0 = "data", cols 1+ = timestamps
    datas = pd.to_datetime(df_raw.iloc[0, 1:].values, errors="coerce")

    # Linhas 1+: col 0 = caminho da tag, cols 1+ = valores
    nomes_brutos = df_raw.iloc[1:, 0].astype(str).tolist()
    dados = df_raw.iloc[1:, 1:].reset_index(drop=True)

    def extrair_partes(caminho):
        return [p.strip() for p in
                caminho.replace("\\\\", "").replace("\\", "/").replace("|", "/").split("/")
                if p.strip()]

    # 1º passo: rótulo = último segmento do caminho
    rotulos = [extrair_partes(n)[-1] if extrair_partes(n) else n for n in nomes_brutos]
    duplicados = {k for k, v in Counter(rotulos).items() if v > 1}

    nomes_unicos = []
    for nome in nomes_brutos:
        partes = extrair_partes(nome)
        rotulo = partes[-1] if partes else nome
        if rotulo in duplicados and len(partes) >= 2:
            sufixo = re.sub(r'[^A-Za-z0-9_]', '', partes[-2])
            nomes_unicos.append(f"{rotulo}_{sufixo}")
        else:
            nomes_unicos.append(rotulo)

    # 2º passo: numeração sequencial para os que ainda persistem
    ainda_dup = {k for k, v in Counter(nomes_unicos).items() if v > 1}
    if ainda_dup:
        ocorrencias = Counter()
        novos = []
        for n in nomes_unicos:
            if n in ainda_dup:
                ocorrencias[n] += 1
                novos.append(f"{n}_{ocorrencias[n]}")
            else:
                novos.append(n)
        nomes_unicos = novos

    # Transpor: (tags × datas) → (datas × tags)
    dados_numericos = dados.apply(pd.to_numeric, errors="coerce").values.T  # shape: (n_datas, n_tags)

    df = pd.DataFrame(data=dados_numericos, index=datas, columns=nomes_unicos)
    df = df[df.index.notna()].sort_index()

    # Remove tags 100% mortas (só zeros ou NaN)
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
    """Aplica filtros APENAS nas tags com regra específica definida pelo engenheiro.
    Todas as demais tags passam sem qualquer alteração (pass-through).
    """
    # Monta dicionário de limites: tag.upper() -> (min, max)
    limites_cfg = parsear_limites_por_variavel(st.session_state.get('limites_texto', ''))
    for regra in st.session_state.get('limites_customizados', []):
        for tag in regra['tags']:
            limites_cfg[tag.upper()] = (regra.get('minimo'), regra.get('maximo'))

    df_saida = pd.DataFrame(index=df_original.index)
    for tag in tags_validas:
        s_orig = pd.to_numeric(df_original[tag], errors="coerce")
        lim_inf, lim_sup = encontrar_limites_para_tag(tag, limites_cfg)

        if lim_inf is not None or lim_sup is not None:
            # Tag COM regra: aplica somente os limites definidos pelo engenheiro
            s = s_orig.copy()
            if lim_inf is not None:
                s = s.mask(s < lim_inf)
            if lim_sup is not None:
                s = s.mask(s > lim_sup)
            df_saida[tag] = s
        else:
            # Tag SEM regra: pass-through, sem nenhuma alteração
            df_saida[tag] = s_orig

    return df_saida


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

    # ----------------------------------------------------------------
    # FLUXO A — uso normal: carregar base já processada (.pkl)
    # ----------------------------------------------------------------
    st.subheader("A. Carregar Base Processada (uso normal)")
    st.caption("Arquivo .pkl gerado anteriormente por este app ou pelo notebook.")

    arq_pkl = st.file_uploader("📥 Base de sensores (.pkl)", type=["pkl"], key="uploader_pkl")

    if arq_pkl:
        with st.spinner("Carregando base..."):
            try:
                df = pd.read_pickle(arq_pkl)
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index, errors="coerce")
                st.session_state["df_pi"] = df
                st.success(f"✅ Base carregada: **{df.shape[0]}** instantes × **{df.shape[1]}** tags.")
                st.dataframe(df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Erro ao carregar PKL: {e}")

    st.divider()

    # ----------------------------------------------------------------
    # FLUXO B — base nova: processar XLSX bruto do PI AF e baixar pkl
    # ----------------------------------------------------------------
    with st.expander("🔄 Processar Nova Base Bruta do PI AF (.xlsx → .pkl)", expanded=False):
        st.caption(
            "Use quando receber uma exportação nova do PI AF.  \n"
            "Formato esperado: linha 0 = timestamps, linhas 1+ = caminhos completos das tags."
        )
        arq_xlsx = st.file_uploader("📥 Exportação bruta do PI AF (.xlsx)", type=["xlsx"], key="uploader_xlsx")

        if arq_xlsx:
            with st.spinner("Processando exportação do PI AF (pode demorar alguns segundos)..."):
                df_novo = preparar_base_pi(arq_xlsx)

            if df_novo is not None:
                st.success(f"✅ Processamento concluído: **{df_novo.shape[0]}** instantes × **{df_novo.shape[1]}** tags ativas.")
                st.dataframe(df_novo.head(5), use_container_width=True)

                # Download do pkl gerado
                import io, pickle
                buf = io.BytesIO()
                pickle.dump(df_novo, buf, protocol=4)
                buf.seek(0)
                nome_saida = arq_xlsx.name.replace(".xlsx", "_limpa.pkl").replace(".XLSX", "_limpa.pkl")
                st.download_button(
                    label="⬇️ Baixar base_pi_limpa.pkl",
                    data=buf,
                    file_name=nome_saida,
                    mime="application/octet-stream",
                    help="Salve este arquivo. Na próxima vez, carregue-o diretamente no Fluxo A."
                )

                if st.button("▶️ Usar esta base agora (sem baixar)", key="btn_usar_agora"):
                    st.session_state["df_pi"] = df_novo
                    st.success("Base carregada na sessão atual!")
                    st.rerun()

    st.divider()

    # ---- Auditoria de Clones Exatos (Célula de Auditoria) ----
    st.subheader("🕵️ Auditoria de Redundância — Sensores Clonados")
    if st.session_state['df_pi'] is None:
        st.info("Carregue a base acima para habilitar a auditoria.")
    else:
        col_audit1, col_audit2 = st.columns(2)
        with col_audit1:
            limiar = st.number_input("Limiar de correlação para redundância", min_value=0.90, max_value=1.0,
                                     value=0.99, step=0.01, key="limiar_corr",
                                     help="Pares com correlação acima deste valor serão sinalizados.")
        with col_audit2:
            st.write("")
            st.write("")
            executar = st.button("🔍 Executar Auditoria Completa", key="btn_auditoria")

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
            if st.checkbox("✅ Confirmo a remoção dos clones.", key="chk_confirma_remocao"):
                if st.button("🗑️ Remover Sensores Redundantes", key="btn_remover_clones"):
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

    # Valida se o arquivo carregado é realmente uma base de sensores
    cols_numericas = df_base.select_dtypes(include=[np.number]).columns.tolist()
    if len(cols_numericas) == 0:
        st.error("❌ O arquivo carregado não parece ser uma base de sensores (nenhuma coluna numérica encontrada).")
        st.info("💡 Certifique-se de carregar a **base de dados PI** no Módulo 1, não a lista de indicadores.")
        st.stop()

    # ---- A. Limites específicos por variável ----
    st.subheader("A. Limites Específicos por Variável (Memória de Limites)")

    with st.expander("➕ Adicionar / Gerenciar Regras de Limite", expanded=False):
        # --- Inicializa estado da busca ---
        if 'lim_cols_encontradas' not in st.session_state:
            st.session_state['lim_cols_encontradas'] = []
        if 'lim_termo_atual' not in st.session_state:
            st.session_state['lim_termo_atual'] = ""

        col_busca, col_btn_busca = st.columns([4, 1])
        termo_lim = col_busca.text_input(
            "1. Digitar termo de busca (ex: potencia, TG11):",
            key="busca_limites",
            placeholder="Digite e clique em Buscar →"
        )
        col_btn_busca.write("")
        col_btn_busca.write("")
        if col_btn_busca.button("🔍 Buscar", key="btn_busca_lim"):
            if termo_lim:
                encontradas = buscar_tags(df_base.columns, termo_lim)
                st.session_state['lim_cols_encontradas'] = encontradas
                st.session_state['lim_termo_atual'] = termo_lim
            else:
                st.session_state['lim_cols_encontradas'] = []

        cols_enc = st.session_state['lim_cols_encontradas']
        if cols_enc:
            termo_atual = st.session_state["lim_termo_atual"]
            st.caption("✅ " + str(len(cols_enc)) + ' tag(s) encontrada(s) para "' + termo_atual + '"')
            tags_sel = st.multiselect("2. Selecionar tags:", cols_enc, key="tags_lim_sel")
            if tags_sel:
                cA, cB = st.columns(2)
                v_min = cA.number_input("Mínimo", value=0.0, key="vmin_lim")
                v_max = cB.number_input("Máximo", value=0.0, key="vmax_lim")
                usar_min = cA.checkbox("Aplicar Mínimo", value=False, key="usar_min_lim")
                usar_max = cB.checkbox("Aplicar Máximo", value=False, key="usar_max_lim")
                if st.button("💾 Salvar Regra", key="btn_salvar_regra"):
                    st.session_state['limites_customizados'].append({
                        'tags': tags_sel,
                        'minimo': v_min if usar_min else None,
                        'maximo': v_max if usar_max else None
                    })
                    # Limpa a busca após salvar
                    st.session_state['lim_cols_encontradas'] = []
                    st.session_state['lim_termo_atual'] = ""
                    st.success(f"✅ Regra salva para {len(tags_sel)} tag(s)!")
                    st.rerun()
        elif st.session_state["lim_termo_atual"]:
            st.warning("Nenhuma tag encontrada para " + repr(st.session_state["lim_termo_atual"]) + ". Tente outro termo.")

    if st.session_state['limites_customizados']:
        st.markdown("**Regras ativas:**")
        df_regras = pd.DataFrame(st.session_state['limites_customizados'])
        st.dataframe(df_regras, use_container_width=True)
        if st.button("🗑️ Limpar Todas as Regras", key="btn_limpar_regras"):
            st.session_state['limites_customizados'] = []
            st.rerun()

    st.divider()

    # ---- B. Aplicar ----
    st.subheader("B. Aplicar Regras")
    st.info(
        "Os filtros são aplicados **somente** nas tags com regra definida acima.  \n"
        "Todas as demais variáveis passam sem qualquer alteração."
    )

    cfg = {}  # Não há parâmetros globais — apenas regras específicas por tag

    # Mostra estado atual da base
    if st.session_state['df_limpo'] is not None:
        n_regras = len(st.session_state.get('limites_customizados', []))
        tags_ja_filtradas = sum(
            1 for regra in st.session_state.get('limites_customizados', [])
            for _ in regra['tags']
        )
        st.success(
            f"✅ Base com filtros ativos: **{tags_ja_filtradas}** tag(s) com regra aplicada.  \n"
            "Adicione mais regras acima e clique em Aplicar para acumular."
        )
        if st.button("↩️ Resetar para base original", key="btn_resetar_base"):
            st.session_state['df_limpo'] = None
            st.session_state['limites_customizados'] = []
            st.success("Base resetada para o estado original.")
            st.rerun()

    col_ap1, col_ap2 = st.columns([2, 3])
    with col_ap1:
        aplicar = st.button("🚀 Aplicar Regras à Base", key="btn_aplicar_limpeza", type="primary")
    with col_ap2:
        st.caption(
            "Aplica as novas regras sobre a base atual (já filtrada).  \n"
            "As regras anteriores são preservadas."
        )

    if aplicar:
        with st.spinner("Aplicando filtros de engenharia..."):
            # Aplica sempre sobre a base original (df_pi) com TODAS as regras acumuladas
            # Isso garante que regras não se sobrepõem de forma inesperada
            base_referencia = st.session_state['df_pi']
            tags_validas = list(base_referencia.columns)
            df_limpo = gerar_dataframe_limpo(base_referencia, tags_validas, cfg)
            st.session_state['df_limpo'] = df_limpo
            st.session_state['cfg_limpeza'] = cfg
            n_tags_regra = sum(len(r['tags']) for r in st.session_state.get('limites_customizados', []))
            st.success(f"✅ Aplicado! **{n_tags_regra}** tag(s) com regra, {len(tags_validas) - n_tags_regra} em pass-through.")

            # Mostra apenas as tags que tiveram regra aplicada
            limites_cfg_ui = {}
            for regra in st.session_state.get('limites_customizados', []):
                for tag in regra['tags']:
                    limites_cfg_ui[tag.upper()] = (regra.get('minimo'), regra.get('maximo'))

            tags_filtradas = [
                tag for tag in tags_validas
                if encontrar_limites_para_tag(tag, limites_cfg_ui)[0] is not None
                or encontrar_limites_para_tag(tag, limites_cfg_ui)[1] is not None
            ]

            if not tags_filtradas:
                st.info("Nenhuma regra definida — base salva sem alterações.")
            else:
                st.write(f"### 📊 Impacto dos Filtros — {len(tags_filtradas)} tag(s) com regra")
                df_antes_num = df_base[tags_filtradas].apply(pd.to_numeric, errors='coerce')
                colunas_stats = ['count', 'mean', '50%', 'std', 'min', 'max']
                try:
                    desc_antes = df_antes_num.describe().T[colunas_stats].rename(columns={'50%': 'mediana'})
                    desc_depois = df_limpo[tags_filtradas].describe().T[colunas_stats].rename(columns={'50%': 'mediana'})
                    comparativo = pd.concat([desc_antes, desc_depois], axis=1, keys=['🔴 ANTES', '🟢 DEPOIS']).round(2)

                    def descr_regra(tag):
                        lims = encontrar_limites_para_tag(tag, limites_cfg_ui)
                        partes = []
                        if lims[0] is not None: partes.append(f"min={lims[0]}")
                        if lims[1] is not None: partes.append(f"max={lims[1]}")
                        return "[" + ", ".join(partes) + "]"

                    comparativo.insert(0, "Regra", [descr_regra(t) for t in comparativo.index])
                    st.dataframe(comparativo, use_container_width=True)
                except Exception:
                    st.warning("Não foi possível gerar tabela comparativa.")

                # Amostra visual da primeira tag filtrada
                st.write(f"### 📈 Amostra Visual — {tags_filtradas[0]}")
                ptag = tags_filtradas[0]
                df_antes_ptag = df_base[ptag].apply(pd.to_numeric) if hasattr(df_base[ptag], 'apply') else pd.to_numeric(df_base[ptag], errors='coerce')
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(df_base.index, pd.to_numeric(df_base[ptag], errors='coerce'),
                        alpha=0.3, label="Original", color='red')
                ax.plot(df_limpo.index, df_limpo[ptag], alpha=0.8, label="Filtrado", color='blue')
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
            help="Col E = Sigla, Col F = Descrição, Col G = Fórmula. Dados a partir da linha 10."
        )

        if arquivo_ind:
            try:
                df_ind = pd.read_excel(arquivo_ind, sheet_name=0, header=None)
                lista_importada = []
                for idx in range(9, len(df_ind)):
                    sigla = str(df_ind.iloc[idx, 4]).strip()
                    if not sigla or sigla == "nan" or "BENCHMARK" in sigla.upper():
                        continue
                    descricao = str(df_ind.iloc[idx, 5]).strip()
                    descricao = "" if descricao == "nan" else descricao
                    formula   = str(df_ind.iloc[idx, 6]).strip()
                    formula   = "" if formula == "nan" else formula
                    lista_importada.append({
                        "sigla":     sigla,
                        "descricao": descricao,
                        "formula":   formula,
                    })

                # Guarda no session_state para sobreviver ao rerender do botão
                st.session_state['_lista_importada'] = lista_importada

                st.success(f"✅ {len(lista_importada)} indicadores encontrados.")
                st.dataframe(
                    pd.DataFrame(lista_importada).rename(columns={
                        "sigla": "Sigla", "descricao": "Descrição", "formula": "Fórmula"
                    }),
                    use_container_width=True,
                    height=250
                )
            except Exception as e:
                st.error(f"Erro ao ler a planilha: {e}")

        # Botão fora do bloco `if arquivo_ind` para funcionar mesmo após rerender
        if st.session_state.get('_lista_importada'):
            n = len(st.session_state['_lista_importada'])
            col_load1, col_load2 = st.columns(2)
            with col_load1:
                btn_carregar = st.button(f"✅ Carregar {n} indicadores (preserva tags já salvas)", key="btn_precarregar")
            with col_load2:
                btn_recarregar = st.button(f"🔄 Recarregar do zero (apaga tags)", key="btn_recarregar")

            if btn_carregar or btn_recarregar:
                for item in st.session_state['_lista_importada']:
                    chave = item['sigla']
                    # Sempre atualiza sigla/descrição/fórmula com os dados novos da planilha
                    # Preserva tags já mapeadas apenas no btn_carregar
                    tags_existentes = st.session_state['mapeamento'].get(chave, {}).get('tags', [])
                    st.session_state['mapeamento'][chave] = {
                        'descricao': item['descricao'],
                        'formula':   item['formula'],
                        'tags':      [] if btn_recarregar else tags_existentes
                    }
                st.session_state['_lista_importada'] = []
                st.success(f"✅ {n} indicador(es) carregado(s) com sucesso!")
                st.rerun()

    # Normaliza dados antigos no mapeamento (garante que todos têm 'descricao' e 'formula' separados)
    for _chave, _item in st.session_state['mapeamento'].items():
        if 'descricao' not in _item:
            _item['descricao'] = ''
        if 'formula' not in _item and 'formula_bruta' in _item:
            _item['formula'] = _item.pop('formula_bruta')
        if 'tags' not in _item:
            _item['tags'] = []

    st.divider()

    # ---- Tabela de Indicadores Mapeados ----
    st.subheader("📋 Indicadores Mapeados")

    if st.session_state['mapeamento']:
        rows = []
        for nome, item in st.session_state['mapeamento'].items():
            status = "✅ Completo" if item.get('tags') else "⏳ Pendente (sem tags)"
            rows.append({
                "Sigla":      nome,
                "Descrição":  item.get('descricao', ''),
                "Fórmula":    item.get('formula', ''),
                "Tags":       ", ".join(item.get('tags', [])),
                "Status":     status
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        col_cl1, col_cl2 = st.columns([1, 4])
        with col_cl1:
            if st.button("🗑️ Limpar Todos", key="btn_limpar_inds"):
                st.session_state['mapeamento'] = {}
                st.rerun()
    else:
        st.info("ℹ️ Nenhum indicador ainda. Importe a lista acima ou crie um novo abaixo.")

    st.divider()

    # ---- Formulário de Criação / Edição ----
    st.subheader("✏️ Vincular Tags a um Indicador")

    nomes_existentes = list(st.session_state['mapeamento'].keys())

    if not nomes_existentes:
        st.info("Importe a lista de indicadores acima primeiro.")
        st.stop()

    # Selectbox para escolher qual indicador editar
    # Inicializa índice selecionado no session_state para persistir entre rerenders
    if '_ind_idx' not in st.session_state:
        st.session_state['_ind_idx'] = 0
    # Ajusta se a lista mudou de tamanho
    if st.session_state['_ind_idx'] >= len(nomes_existentes):
        st.session_state['_ind_idx'] = 0

    nome_edit = st.selectbox(
        "Selecione o indicador para vincular tags:",
        nomes_existentes,
        index=st.session_state['_ind_idx'],
        key="sel_nome_edit"
    )
    # Atualiza índice ao mudar seleção
    st.session_state['_ind_idx'] = nomes_existentes.index(nome_edit)

    item_edit = st.session_state['mapeamento'][nome_edit]

    # Mostra descrição e fórmula do indicador selecionado (somente leitura)
    col_info1, col_info2 = st.columns(2)
    col_info1.info(f"**Descrição:** {item_edit.get('descricao', '—')}")
    col_info2.info(f"**Fórmula:** `{item_edit.get('formula', '—')}`")

    # Chaves de estado com namespace por indicador — evita colisão entre rerenders
    k_cols = f"_map_cols_{nome_edit}"
    k_termo = f"_map_termo_{nome_edit}"
    k_tags = f"_map_tags_{nome_edit}"

    # Inicializa estado para este indicador se ainda não existe
    if k_cols not in st.session_state:
        st.session_state[k_cols] = []
    if k_termo not in st.session_state:
        st.session_state[k_termo] = ""
    if k_tags not in st.session_state:
        # Pré-carrega tags já salvas neste indicador
        st.session_state[k_tags] = list(item_edit.get('tags', []))

    st.caption("💡 Busque e selecione as tags na ordem das variáveis da fórmula. Pode buscar várias vezes para acumular.")

    col_t1, col_t2 = st.columns([4, 1])
    termo_tag = col_t1.text_input(
        "Buscar tags (ex: Potencia, AberturaIGV, TG11):",
        key=f"busca_tags_{nome_edit}",
        placeholder="Digite e clique em Buscar →"
    )
    col_t2.write("")
    col_t2.write("")
    if col_t2.button("🔍 Buscar", key=f"btn_busca_{nome_edit}"):
        if termo_tag:
            encontradas = [c for c in df_base.columns
                           if normalizar(termo_tag) in normalizar(c)
                           and ((df_base[c].isna().sum() + (df_base[c] == 0).sum()) / len(df_base)) < 0.99]
            st.session_state[k_cols] = encontradas
            st.session_state[k_termo] = termo_tag
        else:
            st.session_state[k_cols] = []

    cols_enc = st.session_state[k_cols]
    if cols_enc:
        st.caption("✅ " + str(len(cols_enc)) + ' tag(s) para "' + st.session_state[k_termo] + '" — selecione e clique em Adicionar')
        selecionadas_agora = st.multiselect(
            "Tags encontradas:",
            cols_enc,
            key=f"tags_enc_{nome_edit}"
        )
        if st.button("➕ Adicionar à seleção", key=f"btn_add_{nome_edit}"):
            for t in selecionadas_agora:
                if t not in st.session_state[k_tags]:
                    st.session_state[k_tags].append(t)
            st.session_state[k_cols] = []
            st.rerun()
    elif st.session_state[k_termo] and not cols_enc:
        st.warning("Nenhuma tag encontrada para " + repr(st.session_state[k_termo]) + ". Tente outro termo.")

    # Tags acumuladas — fonte de verdade é k_tags
    tags_atuais = st.session_state[k_tags]
    tags_sel = st.multiselect(
        "🗂️ Tags vinculadas a este indicador (reordene se necessário):",
        options=tags_atuais,
        default=tags_atuais,
        key=f"tags_final_{nome_edit}"
    )
    # Sincroniza remoções feitas no multiselect de volta para k_tags
    if set(tags_sel) != set(tags_atuais) or tags_sel != tags_atuais:
        st.session_state[k_tags] = tags_sel
        st.rerun()

    if tags_atuais:
        if st.button("🗑️ Limpar tags", key=f"btn_clear_{nome_edit}"):
            st.session_state[k_tags] = []
            st.rerun()

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        salvar = st.button("💾 Salvar", type="primary", key="btn_salvar_ind")
    with col_btn2:
        if st.button("🗑️ Remover este Indicador", key="btn_remover_ind"):
            del st.session_state['mapeamento'][nome_edit]
            st.session_state['_map_ind_anterior'] = None
            st.rerun()

    if salvar:
        tags_para_salvar = st.session_state.get(k_tags, [])
        if not tags_para_salvar:
            st.error("Selecione ao menos uma tag antes de salvar.")
        else:
            st.session_state['mapeamento'][nome_edit]['tags'] = tags_para_salvar
            st.success(f"✅ Tags do indicador '{nome_edit}' salvas!")
            # Avança automaticamente para o próximo indicador pendente
            pendentes = [n for n, v in st.session_state['mapeamento'].items() if not v.get('tags')]
            if pendentes:
                prox = pendentes[0]
                st.session_state['_ind_idx'] = nomes_existentes.index(prox)
                st.session_state['_map_ind_anterior'] = None
                st.info(f"➡️ Próximo pendente: **{prox}**")
            st.rerun()


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
        datas = st.date_input("Período de análise:", [d_min, d_max], min_value=d_min, max_value=d_max, key="date_periodo")
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

        descricao_ind = item.get('descricao', '')
        titulo_exp = f"📈 {nome_ind}"
        if descricao_ind:
            titulo_exp += f" — {descricao_ind}"
        if formula:
            titulo_exp += f"  ·  `{formula}`"
        with st.expander(titulo_exp, expanded=True):
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
                cols_pca = buscar_tags(df_avancado.columns, termo_pca)
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
                cols_t2 = buscar_tags(df_avancado.columns, termo_t2)
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
