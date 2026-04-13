# -*- coding: utf-8 -*-
"""
app.py — Interface Streamlit do OPERA Mapeamento.
Toda lógica de cálculo está em core.py.
"""

import io
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core import (
    normalizar,
    buscar_tags,
    preparar_base_pi,
    encontrar_limites_para_tag,
    gerar_dataframe_limpo,
    extrair_vars_formula,
    calcular_indicador,
    executar_pca,
    calcular_t2,
)

import re

# ==========================================
# 0. CONFIGURAÇÃO E ESTADO
# ==========================================
st.set_page_config(page_title="OPERA - Process Control", layout="wide", page_icon="🏭")

defaults = {
    'df_pi': None,
    'df_limpo': None,
    'mapeamento': {},
    'limites_customizados': [],
    'limites_texto': "",
    'lixo_para_remover': [],
    'msg_sucesso': None,
    'pca_n_top': 10,
    '_lista_importada': [],
    '_pca_cols': [],
    '_pca_termo': "",
    '_t2_cols': [],
    '_t2_termo': "",
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
                mascara_original = ~df_audit.T.duplicated(keep='first')
                mascara_clone    =  df_audit.T.duplicated(keep='first')
                tags_clonadas = df_audit.columns[mascara_clone].tolist()

                if tags_clonadas:
                    st.warning(f"🚨 {len(tags_clonadas)} variável(is) são cópias exatas de outro sensor.")

                    # Para cada clone, identifica o original correspondente
                    rows_clone = []
                    for clone in tags_clonadas:
                        # Compara com todas as tags originais para achar a fonte
                        for orig in df_audit.columns[mascara_original]:
                            if df_audit[clone].equals(df_audit[orig]):
                                rows_clone.append({"Original (manter)": orig, "Clone (remover)": clone})
                                break
                        else:
                            rows_clone.append({"Original (manter)": "—", "Clone (remover)": clone})

                    st.dataframe(pd.DataFrame(rows_clone), use_container_width=True, hide_index=True)

                    # Adiciona clones à lista de remoção
                    if tags_clonadas not in [st.session_state.get('lixo_para_remover', [])]:
                        if st.button("🗑️ Marcar clones para remoção", key="btn_marcar_clones"):
                            st.session_state['lixo_para_remover'] = list(
                                set(st.session_state.get('lixo_para_remover', []) + tags_clonadas)
                            )
                            st.success(f"{len(tags_clonadas)} clone(s) marcado(s) para remoção.")
                            st.rerun()
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
            # Alterado para usar as variáveis locais corretas (cols_enc e termo_atual)
            st.caption(f"✅ {len(cols_enc)} tag(s) encontrada(s) para \"{termo_atual}\"")
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
            df_limpo = gerar_dataframe_limpo(
                base_referencia, tags_validas,
                st.session_state.get("limites_customizados", []),
                st.session_state.get("limites_texto", ""))
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
    formula_edit = item_edit.get('formula', '')

    # Mostra descrição e fórmula do indicador selecionado (somente leitura)
    col_info1, col_info2 = st.columns(2)
    col_info1.info(f"**Descrição:** {item_edit.get('descricao', '—')}")
    col_info2.info(f"**Fórmula:** `{formula_edit or '—'}`")

    # ── Extrai variáveis da fórmula (mesma lógica do algoritmo original) ──
    TERMOS_IGNORAR = {'SOMA','MEDIA','SE','MAX','MIN','IF','AND','OR','NAN'}
    parte_dir = formula_edit.split("=", 1)[1] if "=" in formula_edit else formula_edit
    palavras = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*%?', parte_dir)
    vars_formula = list(dict.fromkeys(
        p for p in palavras if p.upper() not in TERMOS_IGNORAR
    )) or ["VAR_UNICA"]

    # ── Estado por indicador (namespace) ──
    k_var_tags = f"_vtags_{nome_edit}"   # dict: var_nome -> [tag, ...]
    k_cols     = f"_cols_{nome_edit}"    # lista de tags encontradas na busca atual
    k_termo    = f"_termo_{nome_edit}"   # termo buscado
    k_var_ativa= f"_varativa_{nome_edit}"# variável que está recebendo tags agora

    # Inicializa se não existir (ou restaura do mapeamento salvo)
    if k_var_tags not in st.session_state:
        # Restaura mapeamento var→tags já salvo, se houver
        salvo = item_edit.get('var_tags', {})
        if salvo:
            st.session_state[k_var_tags] = {v: list(t) for v, t in salvo.items()}
        else:
            st.session_state[k_var_tags] = {v: [] for v in vars_formula}
    if k_cols not in st.session_state:
        st.session_state[k_cols] = []
    if k_termo not in st.session_state:
        st.session_state[k_termo] = ""
    if k_var_ativa not in st.session_state:
        st.session_state[k_var_ativa] = vars_formula[0]

    var_tags = st.session_state[k_var_tags]

    # Garante que novas variáveis extraídas da fórmula estejam no dict
    for v in vars_formula:
        if v not in var_tags:
            var_tags[v] = []

    st.markdown("#### 🔗 Vinculação Variável → Tag(s) PI")
    st.caption("Para cada variável da fórmula, busque as tags correspondentes. Se houver múltiplas unidades (ex: TG11 e TG12), adicione uma tag por unidade para cada variável — o cálculo roda em paralelo.")

    # ── Tabela de progresso das variáveis ──
    prog_rows = []
    for v in vars_formula:
        tags_v = var_tags.get(v, [])
        prog_rows.append({
            "Variável": v,
            "Tags vinculadas": ", ".join(tags_v) if tags_v else "—",
            "N": len(tags_v)
        })
    st.dataframe(pd.DataFrame(prog_rows), use_container_width=True, hide_index=True)

    # ── Seletor de variável ativa ──
    var_ativa = st.selectbox(
        "Variável a vincular agora:",
        vars_formula,
        index=vars_formula.index(st.session_state[k_var_ativa])
              if st.session_state[k_var_ativa] in vars_formula else 0,
        key=f"sel_var_ativa_{nome_edit}"
    )
    st.session_state[k_var_ativa] = var_ativa

    # ── Busca de tags para a variável ativa ──
    col_t1, col_t2 = st.columns([4, 1])
    termo_tag = col_t1.text_input(
        f"Buscar tags para **{var_ativa}**:",
        key=f"busca_{nome_edit}_{var_ativa}",
        placeholder="Digite e clique em Buscar →"
    )
    col_t2.write("")
    col_t2.write("")
    if col_t2.button("🔍 Buscar", key=f"btn_busca_{nome_edit}_{var_ativa}"):
        if termo_tag:
            encontradas = [
                c for c in df_base.columns
                if normalizar(termo_tag) in normalizar(c)
                and ((df_base[c].isna().sum() + (df_base[c] == 0).sum()) / len(df_base)) < 0.99
            ]
            st.session_state[k_cols] = encontradas
            st.session_state[k_termo] = termo_tag
        else:
            st.session_state[k_cols] = []
            st.session_state[k_termo] = ""

    cols_enc = st.session_state[k_cols]
    if cols_enc:
        st.caption(f"✅ {len(cols_enc)} tag(s) encontrada(s) — selecione e clique em Adicionar")
        selecionadas = st.multiselect(
            "Tags encontradas:",
            cols_enc,
            key=f"tags_enc_{nome_edit}_{var_ativa}"
        )
        if st.button(f"➕ Adicionar a [{var_ativa}]", key=f"btn_add_{nome_edit}_{var_ativa}"):
            for t in selecionadas:
                if t not in var_tags[var_ativa]:
                    var_tags[var_ativa].append(t)
            st.session_state[k_cols] = []
            st.session_state[k_termo] = ""
            # Avança para próxima variável sem tags
            sem_tags = [v for v in vars_formula if not var_tags.get(v)]
            if sem_tags and sem_tags[0] != var_ativa:
                st.session_state[k_var_ativa] = sem_tags[0]
            st.rerun()
    elif st.session_state[k_termo]:
        st.warning(f"Nenhuma tag encontrada para '{st.session_state[k_termo]}'. Tente outro termo.")

    # ── Tags já vinculadas à variável ativa (editável) ──
    tags_var_atual = var_tags.get(var_ativa, [])
    if tags_var_atual:
        st.caption(f"Tags de **{var_ativa}** (clique para remover):")
        tags_editadas = st.multiselect(
            f"Tags de {var_ativa}:",
            options=tags_var_atual,
            default=tags_var_atual,
            key=f"tags_edit_{nome_edit}_{var_ativa}",
            label_visibility="collapsed"
        )
        if tags_editadas != tags_var_atual:
            var_tags[var_ativa] = tags_editadas
            st.rerun()
        if st.button(f"🗑️ Limpar tags de [{var_ativa}]", key=f"btn_clear_{nome_edit}_{var_ativa}"):
            var_tags[var_ativa] = []
            st.rerun()

    st.divider()

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        salvar = st.button("💾 Salvar", type="primary", key="btn_salvar_ind")
    with col_btn2:
        if st.button("🗑️ Remover este Indicador", key="btn_remover_ind"):
            del st.session_state['mapeamento'][nome_edit]
            st.rerun()

    if salvar:
        # Valida: todas as variáveis precisam ter ao menos 1 tag
        sem_tag = [v for v in vars_formula if not var_tags.get(v)]
        if sem_tag:
            st.error(f"Variáveis sem tag: **{', '.join(sem_tag)}**. Vincule ao menos 1 tag para cada.")
        else:
            # Verifica consistência: mesmo N de tags por variável (para o chunk funcionar)
            ns = [len(var_tags[v]) for v in vars_formula]
            if len(set(ns)) > 1:
                st.warning(
                    f"⚠️ As variáveis têm números diferentes de tags: "
                    + ", ".join(f"{v}={len(var_tags[v])}" for v in vars_formula)
                    + ". O cálculo usará o menor conjunto."
                )
            # Salva: var_tags (dict) e tags (lista flat, para compatibilidade)
            tags_flat = []
            for v in vars_formula:
                tags_flat.extend(var_tags.get(v, []))
            st.session_state['mapeamento'][nome_edit]['var_tags'] = dict(var_tags)
            st.session_state['mapeamento'][nome_edit]['tags'] = tags_flat
            st.session_state['mapeamento'][nome_edit]['vars_formula'] = vars_formula
            st.success(f"✅ '{nome_edit}' salvo!")
            # Avança para próximo pendente
            pendentes = [n for n, v in st.session_state['mapeamento'].items()
                         if not v.get('var_tags') or any(not vt for vt in v['var_tags'].values())]
            if pendentes and pendentes[0] != nome_edit:
                st.session_state['_ind_idx'] = nomes_existentes.index(pendentes[0])
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

            var_tags_map = item.get("var_tags")
            resultados, erro = calcular_indicador(df_target, formula, tags_validas, var_tags=var_tags_map)

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

        # Estado persistente da busca PCA
        if '_pca_cols' not in st.session_state:
            st.session_state['_pca_cols'] = []
        if '_pca_termo' not in st.session_state:
            st.session_state['_pca_termo'] = ""

        col_p1, col_p2 = st.columns([4, 1])
        termo_pca = col_p1.text_input("Filtrar tags para PCA (ex: TG11, Vazao):", key="pca_input",
                                       placeholder="Digite e clique em Buscar →")
        col_p2.write("")
        col_p2.write("")
        if col_p2.button("🔍 Buscar", key="btn_pca_buscar"):
            if termo_pca:
                encontradas = buscar_tags(df_avancado.columns, termo_pca)
                st.session_state['_pca_cols'] = encontradas
                st.session_state['_pca_termo'] = termo_pca
            else:
                st.session_state['_pca_cols'] = []

        cols_pca_encontradas = st.session_state['_pca_cols']
        if cols_pca_encontradas:
            termo_atual_pca = st.session_state["_pca_termo"]
            st.caption(f"\u2705 {len(cols_pca_encontradas)} tag(s) encontrada(s) para \"{termo_atual_pca}\"")
            cols_pca_sel = st.multiselect(
                "Tags para o PCA (desmarque as que não quer incluir):",
                cols_pca_encontradas,
                default=cols_pca_encontradas,
                key="pca_multisel"
            )
        elif st.session_state['_pca_termo']:
            st.warning("Nenhuma tag encontrada para \"" + st.session_state["_pca_termo"] + "\". Tente outro termo.")
            cols_pca_sel = []
        else:
            cols_pca_sel = []

        n_top = st.number_input("Top N sensores nos Loadings:", min_value=5, max_value=50, value=10, key="pca_top")

        if st.button("🚀 Processar PCA", key="btn_pca"):
            if not cols_pca_sel:
                st.error("Busque e selecione as tags antes de processar.")
            elif len(cols_pca_sel) < 2:
                st.error(f"PCA precisa de ao menos 2 tags. Selecione mais.")
            else:
                cols_pca = cols_pca_sel
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

        # Estado persistente da busca T²
        if '_t2_cols' not in st.session_state:
            st.session_state['_t2_cols'] = []
        if '_t2_termo' not in st.session_state:
            st.session_state['_t2_termo'] = ""

        col_t1, col_t2b = st.columns([4, 1])
        termo_t2 = col_t1.text_input("Agrupar sensores para T² (ex: Turbina, TG11):", key="t2_input",
                                      placeholder="Digite e clique em Buscar →")
        col_t2b.write("")
        col_t2b.write("")
        if col_t2b.button("🔍 Buscar", key="btn_t2_buscar"):
            if termo_t2:
                encontradas = buscar_tags(df_avancado.columns, termo_t2)
                st.session_state['_t2_cols'] = encontradas
                st.session_state['_t2_termo'] = termo_t2
            else:
                st.session_state['_t2_cols'] = []

        cols_t2_encontradas = st.session_state['_t2_cols']
        if cols_t2_encontradas:
            termo_atual_t2 = st.session_state["_t2_termo"]
            st.caption(f"\u2705 {len(cols_t2_encontradas)} tag(s) encontrada(s) para \"{termo_atual_t2}\"")
            cols_t2_sel = st.multiselect(
                "Tags para o T² (desmarque as que não quer incluir):",
                cols_t2_encontradas,
                default=cols_t2_encontradas,
                key="t2_multisel"
            )
        elif st.session_state['_t2_termo']:
            st.warning("Nenhuma tag encontrada para \"" + st.session_state["_t2_termo"] + "\". Tente outro termo.")
            cols_t2_sel = []
        else:
            cols_t2_sel = []

        confianca = st.slider("Confiança Estatística", 0.90, 0.999, 0.99, 0.001, key="t2_conf")

        if st.button("🚀 Calcular Carta T²", key="btn_t2"):
            if not cols_t2_sel:
                st.error("Busque e selecione as tags antes de calcular.")
            elif len(cols_t2_sel) < 2:
                st.error("O T² exige ao menos 2 variáveis.")
            else:
                cols_t2 = cols_t2_sel
                with st.spinner("Calculando Covariância..."):
                    try:
                        t2_valores, lsc_t2, df_clean, var_principal = calcular_t2(df_avancado, cols_t2, confianca)
                    except ValueError as e:
                        st.error(f"❌ {e}")
                        t2_valores = None

                if t2_valores is not None:
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
