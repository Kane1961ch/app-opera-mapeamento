# -*- coding: utf-8 -*-
"""
core.py — Funções de backend do OPERA Mapeamento.
Sem dependência de Streamlit: pode ser importado e testado de forma independente.
"""

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
# UTILITÁRIOS
# ──────────────────────────────────────────────

def normalizar(texto: str) -> str:
    """Remove acentos e converte para minúsculas (busca tolerante)."""
    return (
        unicodedata.normalize("NFD", str(texto))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )


def buscar_tags(colunas, termo: str) -> list:
    """Filtra colunas cujo nome normalizado contém o termo normalizado."""
    t = normalizar(termo)
    return [c for c in colunas if t in normalizar(c)]


# ──────────────────────────────────────────────
# PROCESSADOR DA BASE PI AF
# ──────────────────────────────────────────────

def preparar_base_pi(arquivo_upload) -> pd.DataFrame:
    """
    Lê exportação bruta do PI AF:
      Linha 0: "data" | timestamp1 | timestamp2 | ...
      Linha 1+: caminho_completo_tag | valor1 | valor2 | ...

    Retorna DataFrame (DatetimeIndex × tags limpas) ou levanta ValueError.
    """
    if hasattr(arquivo_upload, "seek"):
        arquivo_upload.seek(0)
    try:
        df_raw = pd.read_excel(arquivo_upload, header=None)
    except Exception as e:
        raise ValueError(f"Erro ao ler o Excel: {e}") from e

    datas = pd.to_datetime(df_raw.iloc[0, 1:].values, errors="coerce")
    nomes_brutos = df_raw.iloc[1:, 0].astype(str).tolist()
    dados = df_raw.iloc[1:, 1:].reset_index(drop=True)

    def extrair_partes(caminho):
        return [p.strip() for p in
                caminho.replace("\\\\", "").replace("\\", "/").replace("|", "/").split("/")
                if p.strip()]

    rotulos = [extrair_partes(n)[-1] if extrair_partes(n) else n for n in nomes_brutos]
    duplicados = {k for k, v in Counter(rotulos).items() if v > 1}

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
        ocorrencias: Counter = Counter()
        novos = []
        for n in nomes_unicos:
            if n in ainda_dup:
                ocorrencias[n] += 1
                novos.append(f"{n}_{ocorrencias[n]}")
            else:
                novos.append(n)
        nomes_unicos = novos

    dados_num = dados.apply(pd.to_numeric, errors="coerce").values.T
    df = pd.DataFrame(data=dados_num, index=datas, columns=nomes_unicos)
    df = df[df.index.notna()].sort_index()
    df = df.loc[:, ~((df == 0) | df.isna()).all(axis=0)]
    return df


# ──────────────────────────────────────────────
# MOTOR HEURÍSTICO DE LIMPEZA
# ──────────────────────────────────────────────

def parsear_limites_por_variavel(texto_limites: str) -> dict:
    """
    Converte string 'TAG1, TAG2 [min,max]; TAG3 [,max]' → dict {TAG: (min, max)}.
    """
    resultado = {}
    texto = str(texto_limites).strip()
    if not texto:
        return resultado
    for bloco in texto.split("]"):
        if not bloco.strip() or "[" not in bloco:
            continue
        nomes_str, limites_str = bloco.split("[", 1)
        lista_nomes = [n.strip().upper() for n in nomes_str.replace(";", ",").split(",") if n.strip()]
        limites = limites_str.split(",")
        if len(limites) >= 2:
            try:
                inf = float(limites[0].strip()) if limites[0].strip() else None
            except ValueError:
                inf = None
            try:
                sup = float(limites[1].strip()) if limites[1].strip() else None
            except ValueError:
                sup = None
            for nome in lista_nomes:
                resultado[nome] = (inf, sup)
    return resultado


def encontrar_limites_para_tag(tag: str, limites_cfg: dict) -> tuple:
    """Busca limites de uma tag no dict, com fallback por substring."""
    tag_upper = str(tag).upper()
    if tag_upper in limites_cfg:
        return limites_cfg[tag_upper]
    for chave, lims in limites_cfg.items():
        if chave in tag_upper:
            return lims
    return (None, None)


def limpar_serie(
    serie: pd.Series,
    remover_zeros: bool,
    remover_negativos: bool,
    lim_inf,
    lim_sup,
    usar_outlier: bool,
    metodo_outlier: str,
    fator_iqr: float,
    limite_zscore: float,
    preencher_ultimo: bool,
    usar_rolling: bool,
    janela_rolling: int,
    pct_minimo: float,
) -> pd.Series:
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


def gerar_dataframe_limpo(
    df_original: pd.DataFrame,
    tags_validas: list,
    limites_customizados: list,
    limites_texto: str = "",
) -> pd.DataFrame:
    """
    Aplica filtros apenas nas tags com regra definida.
    Tags sem regra passam sem alteração (pass-through).

    limites_customizados: lista de dicts {'tags': [...], 'minimo': x, 'maximo': y}
    limites_texto: string legado no formato 'TAG [min,max]'
    """
    limites_cfg = parsear_limites_por_variavel(limites_texto)
    for regra in limites_customizados:
        for tag in regra.get("tags", []):
            limites_cfg[tag.upper()] = (regra.get("minimo"), regra.get("maximo"))

    df_saida = pd.DataFrame(index=df_original.index)
    for tag in tags_validas:
        s_orig = pd.to_numeric(df_original[tag], errors="coerce")
        lim_inf, lim_sup = encontrar_limites_para_tag(tag, limites_cfg)
        if lim_inf is not None or lim_sup is not None:
            s = s_orig.copy()
            if lim_inf is not None:
                s = s.mask(s < lim_inf)
            if lim_sup is not None:
                s = s.mask(s > lim_sup)
            df_saida[tag] = s
        else:
            df_saida[tag] = s_orig
    return df_saida


# ──────────────────────────────────────────────
# MOTOR CALCULADOR DE INDICADORES
# ──────────────────────────────────────────────

TERMOS_IGNORAR = {"SOMA", "MEDIA", "SE", "MAX", "MIN", "IF", "AND", "OR", "NAN"}


def extrair_vars_formula(formula: str) -> list:
    """Extrai variáveis únicas da parte direita de uma fórmula."""
    parte = formula.split("=", 1)[1] if "=" in formula else formula
    palavras = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*%?", parte)
    return list(dict.fromkeys(p for p in palavras if p.upper() not in TERMOS_IGNORAR)) or ["VAR_UNICA"]


def calcular_indicador(
    df_plot: pd.DataFrame,
    formula: str,
    tags_validas: list,
    var_tags: dict = None,
) -> tuple:
    """
    Calcula série temporal do indicador.

    var_tags: {var_nome: [tag_unid1, tag_unid2, ...]} — mapeamento explícito.
              Se None, distribui tags_validas igualmente pelas variáveis (modo legado).

    Retorna (dict {nome_unidade: pd.Series}, erro_str | None).
    """
    vars_formula = extrair_vars_formula(formula)
    parte_direita = (formula.split("=", 1)[1] if "=" in formula else formula).strip()

    if var_tags:
        mapping = {v: [t for t in var_tags.get(v, []) if t in df_plot.columns]
                   for v in vars_formula}
        ns = [len(mapping[v]) for v in vars_formula if mapping.get(v)]
        chunk = min(ns) if ns else 0
    else:
        if not vars_formula or len(tags_validas) % len(vars_formula) != 0:
            return None, "Número de tags incompatível com a fórmula."
        chunk = len(tags_validas) // len(vars_formula)
        mapping = {var: tags_validas[i * chunk:(i + 1) * chunk]
                   for i, var in enumerate(vars_formula)}

    if chunk == 0:
        return None, "Nenhuma tag válida mapeada para as variáveis."

    resultados = {}
    vars_ordenadas = sorted(vars_formula, key=len, reverse=True)

    for k in range(chunk):
        tag_principal = mapping[vars_formula[0]][k]
        partes = tag_principal.rsplit("_", 1)
        maq = partes[-1] if len(partes) > 1 else f"Unid_{k + 1}"

        df_eval = pd.DataFrame(index=df_plot.index)
        expressao = parte_direita
        for var in vars_ordenadas:
            tags_var = mapping.get(var, [])
            if k >= len(tags_var):
                continue
            safe_var = re.sub(r"[^a-zA-Z0-9_]", "", var) or "VAR_SYMB"
            df_eval[safe_var] = pd.to_numeric(df_plot[tags_var[k]], errors="coerce")
            expressao = re.sub(rf"\b{re.escape(var)}\b", safe_var, expressao)

        try:
            serie = df_eval.eval(expressao, engine="python")
        except Exception:
            try:
                local_dict = {col: df_eval[col] for col in df_eval.columns}
                serie = eval(expressao, {"__builtins__": None, "np": np, "pd": pd}, local_dict)
            except Exception as e:
                return None, str(e)

        serie = pd.Series(serie, index=df_plot.index).replace([np.inf, -np.inf], np.nan)
        resultados[maq] = serie

    return (resultados, None) if resultados else (None, "Nenhum resultado calculado.")


# ──────────────────────────────────────────────
# PCA
# ──────────────────────────────────────────────

def executar_pca(df: pd.DataFrame, cols: list) -> tuple:
    """
    Retorna (pca_obj, pca_data_array, df_colunas_validas).
    Remove colunas estáticas (std ≈ 0) e linhas com NaN.
    """
    df_temp = df[cols].apply(pd.to_numeric, errors="coerce")
    df_temp = df_temp.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    cols_validas = df_temp.columns[df_temp.std() > 1e-6]
    df_final = df_temp[cols_validas]
    X_scaled = StandardScaler().fit_transform(df_final)
    pca = PCA()
    pca_data = pca.fit_transform(X_scaled)
    return pca, pca_data, df_final


# ──────────────────────────────────────────────
# T² DE HOTELLING
# ──────────────────────────────────────────────

def calcular_t2(df: pd.DataFrame, cols: list, nivel_confianca: float) -> tuple:
    """
    Retorna (t2_valores, lsc_t2, df_clean, var_principal) ou levanta ValueError.
    """
    df_clean = df[cols].apply(pd.to_numeric, errors="coerce")
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    df_clean = df_clean.loc[:, df_clean.std() > 1e-6]

    if df_clean.shape[1] < 2:
        raise ValueError("Menos de 2 variáveis após limpeza.")

    media_vec = df_clean.mean()
    cov_matriz = df_clean.cov()
    try:
        inv_cov = np.linalg.inv(cov_matriz)
    except np.linalg.LinAlgError:
        raise ValueError("Matriz de covariância singular (variáveis perfeitamente colineares).")

    diff = df_clean - media_vec
    t2_valores = np.sum((np.dot(diff, inv_cov)) * diff, axis=1)
    lsc_t2 = chi2.ppf(nivel_confianca, df=df_clean.shape[1])
    var_principal = df_clean.var().idxmax()
    return t2_valores, lsc_t2, df_clean, var_principal
