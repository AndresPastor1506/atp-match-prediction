# ==============================================================================
# viz_atp.py
# Módulo de visualización interactiva — Circuito ATP (1968–2024)
# Requiere: pandas, numpy, matplotlib, ipywidgets
# ==============================================================================

import glob
import os
import warnings

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# CONSTANTES
# ------------------------------------------------------------------------------
COLORES_SUPERFICIE = {
    "Hard":   "#2196F3",
    "Clay":   "#E64A19",
    "Grass":  "#388E3C",
    "Carpet": "#7B1FA2",
}
SUPERFICIES_MAIN = ["Hard", "Clay", "Grass"]
AÑO_MIN_GLOBAL = 1968
AÑO_MAX_GLOBAL = 2024

plt.rcParams.update({
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor":   "#ffffff",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size": 10,
})


# ==============================================================================
# 1. CARGA Y PREPROCESAMIENTO
# ==============================================================================

def cargar_datos(ruta_repo: str = "tennis_atp") -> pd.DataFrame:
    """
    Carga y consolida todos los CSVs de partidos ATP principales (1968–2024).
    Excluye futures, qualys, challengers y dobles.
    Aplica limpieza básica y filtra al rango 1968–2024.

    Parámetros
    ----------
    ruta_repo : str
        Ruta al repositorio clonado de Jeff Sackmann.

    Retorna
    -------
    pd.DataFrame
        Dataset limpio listo para visualización.
    """
    EXCLUIR = ("futures", "qual", "chall", "doubles")
    todos_csv = glob.glob(f"{ruta_repo}/atp_matches_*.csv")
    archivos = [
        f for f in todos_csv
        if not any(ex in os.path.basename(f) for ex in EXCLUIR)
    ]
    archivos.sort()

    df = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in archivos],
        ignore_index=True,
    )

    # --- Columna de año ---
    df["year"] = df["tourney_date"].astype(str).str[:4].astype(int, errors="ignore")
    df = df[df["year"].between(AÑO_MIN_GLOBAL, AÑO_MAX_GLOBAL)].copy()

    # --- Superficie ---
    df["surface"] = df["surface"].fillna("Unknown")

    # --- Métricas derivadas ---
    df["total_aces"]   = df["w_ace"].fillna(0) + df["l_ace"].fillna(0)
    df["w_1st_pct"]    = df["w_1stIn"] / df["w_svpt"] * 100
    df["l_1st_pct"]    = df["l_1stIn"] / df["l_svpt"] * 100

    # --- Limpieza de nombres ---
    df["winner_name"] = df["winner_name"].str.strip()
    df["loser_name"]  = df["loser_name"].str.strip()

    print(f"✅ Dataset cargado: {len(df):,} partidos | "
          f"{int(df['year'].min())}–{int(df['year'].max())}")
    print(f"   Superficies disponibles: {sorted(df['surface'].unique().tolist())}")

    return df


def _filtrar(df: pd.DataFrame, años: tuple, superficie: str) -> pd.DataFrame:
    """Filtra el DataFrame por rango de años y superficie."""
    mask = df["year"].between(años[0], años[1])
    if superficie != "Todas":
        mask &= df["surface"] == superficie
    return df[mask].copy()


# ==============================================================================
# 2. PÁGINA 1A — KPIs + Gráfico de anillos
# ==============================================================================

def _render_kpis_anillo(df_raw: pd.DataFrame, años: tuple, superficie: str,
                        out: widgets.Output) -> None:
    df = _filtrar(df_raw, años, superficie)

    total_partidos    = len(df)
    jugadores_unicos  = len(set(df["winner_name"].dropna()) | set(df["loser_name"].dropna()))
    duracion_media    = df["minutes"].mean()
    año_mas_partidos  = df.groupby("year").size().idxmax() if total_partidos > 0 else "N/A"

    with out:
        clear_output(wait=True)

        # KPIs
        fig, axes = plt.subplots(1, 4, figsize=(14, 2.2))
        fig.patch.set_facecolor("#f0f4f8")
        kpis = [
            ("Total partidos",    f"{total_partidos:,}"),
            ("Jugadores únicos",  f"{jugadores_unicos:,}"),
            ("Duración media",    f"{duracion_media:.0f} min" if pd.notna(duracion_media) else "Sin datos"),
            ("Año más activo",    str(año_mas_partidos)),
        ]
        for ax, (titulo, valor) in zip(axes, kpis):
            ax.set_facecolor("#ffffff")
            ax.text(0.5, 0.65, valor, ha="center", va="center",
                    fontsize=20, fontweight="bold", color="#1a1a2e", transform=ax.transAxes)
            ax.text(0.5, 0.22, titulo, ha="center", va="center",
                    fontsize=10, color="#555555", transform=ax.transAxes)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

        # Anillo
        conteo_sup = df["surface"].value_counts()
        conteo_sup = conteo_sup[conteo_sup.index != "Unknown"]
        colores = [COLORES_SUPERFICIE.get(s, "#999999") for s in conteo_sup.index]

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        wedges, texts, autotexts = ax2.pie(
            conteo_sup.values,
            labels=conteo_sup.index,
            autopct="%1.1f%%",
            colors=colores,
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax2.set_title("Distribución de partidos por superficie", fontsize=12, pad=15)
        plt.tight_layout()
        plt.show()


def pagina1_kpis(df_raw: pd.DataFrame) -> None:
    """Página 1A — KPIs globales y gráfico de anillos por superficie."""
    año_min = int(df_raw["year"].min())
    año_max = int(df_raw["year"].max())
    superficies = ["Todas"] + sorted([s for s in df_raw["surface"].unique() if s != "Unknown"])

    w_años = widgets.IntRangeSlider(
        value=[año_min, año_max], min=año_min, max=año_max, step=1,
        description="Años:", continuous_update=False,
        layout=widgets.Layout(width="500px"),
    )
    w_sup = widgets.Dropdown(options=superficies, value="Todas", description="Superficie:")
    out   = widgets.Output()

    def on_change(_):
        _render_kpis_anillo(df_raw, w_años.value, w_sup.value, out)

    w_años.observe(on_change, names="value")
    w_sup.observe(on_change, names="value")

    display(widgets.HBox([w_años, w_sup]), out)
    _render_kpis_anillo(df_raw, w_años.value, w_sup.value, out)


# ==============================================================================
# 3. PÁGINA 1B — Top 15 victorias + Top 10 aces
# ==============================================================================

def _render_rankings(df_raw: pd.DataFrame, años: tuple, superficie: str,
                     out: widgets.Output) -> None:
    df = _filtrar(df_raw, años, superficie)

    top_wins = df["winner_name"].value_counts().head(15).sort_values(ascending=True)

    aces_w   = df.groupby("winner_name")["w_ace"].sum()
    aces_l   = df.groupby("loser_name")["l_ace"].sum()
    aces_tot = aces_w.add(aces_l, fill_value=0).sort_values(ascending=False).head(10).sort_values(ascending=True)

    with out:
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Victorias
        bars1 = ax1.barh(top_wins.index, top_wins.values, color="#1565C0", edgecolor="white")
        ax1.set_title("Top 15 jugadores por victorias", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Victorias")
        for bar in bars1:
            ax1.text(bar.get_width() + max(top_wins.values) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"{int(bar.get_width()):,}", va="center", fontsize=8, color="#333")
        ax1.set_xlim(0, max(top_wins.values) * 1.12)

        # Aces
        bars2 = ax2.barh(aces_tot.index, aces_tot.values, color="#00796B", edgecolor="white")
        ax2.set_title("Top 10 jugadores por aces totales", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Aces")
        for bar in bars2:
            ax2.text(bar.get_width() + max(aces_tot.values) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"{int(bar.get_width()):,}", va="center", fontsize=8, color="#333")
        ax2.set_xlim(0, max(aces_tot.values) * 1.12)

        plt.tight_layout()
        plt.show()


def pagina1_rankings(df_raw: pd.DataFrame) -> None:
    """Página 1B — Top 15 victorias y Top 10 en aces."""
    año_min = int(df_raw["year"].min())
    año_max = int(df_raw["year"].max())
    superficies = ["Todas"] + sorted([s for s in df_raw["surface"].unique() if s != "Unknown"])

    w_años = widgets.IntRangeSlider(
        value=[año_min, año_max], min=año_min, max=año_max, step=1,
        description="Años:", continuous_update=False,
        layout=widgets.Layout(width="500px"),
    )
    w_sup = widgets.Dropdown(options=superficies, value="Todas", description="Superficie:")
    out   = widgets.Output()

    def on_change(_):
        _render_rankings(df_raw, w_años.value, w_sup.value, out)

    w_años.observe(on_change, names="value")
    w_sup.observe(on_change, names="value")

    display(widgets.HBox([w_años, w_sup]), out)
    _render_rankings(df_raw, w_años.value, w_sup.value, out)


# ==============================================================================
# 4. PÁGINA 1C — Finales Grand Slam + Upsets
# ==============================================================================

def _render_gs(df_raw: pd.DataFrame, años: tuple, out: widgets.Output) -> None:
    df = df_raw[
        df_raw["year"].between(años[0], años[1]) &
        (df_raw["tourney_level"] == "G") &
        (df_raw["round"] == "F")
    ].copy()

    tabla = df[["year", "tourney_name", "winner_name", "loser_name", "score", "surface"]].copy()
    tabla.columns = ["Año", "Torneo", "Ganador", "Finalista", "Score", "Superficie"]
    tabla = tabla.sort_values("Año", ascending=False).reset_index(drop=True)

    with out:
        clear_output(wait=True)
        print(f"\n🏆 FINALES DE GRAND SLAM ({años[0]}–{años[1]}) — {len(tabla)} finales\n")
        print(tabla.to_string(index=False))


def _render_upsets(df_raw: pd.DataFrame, años: tuple, gap: int,
                   out: widgets.Output) -> None:
    df = df_raw[
        df_raw["year"].between(años[0], años[1]) &
        df_raw["winner_rank"].notna() &
        df_raw["loser_rank"].notna()
    ].copy()

    df["rank_gap"] = df["winner_rank"] - df["loser_rank"]
    upsets = df[df["rank_gap"] >= gap].sort_values("rank_gap", ascending=False)

    tabla = upsets[["winner_name", "loser_name", "winner_rank", "loser_rank",
                    "tourney_name", "round", "score", "year"]].head(30).copy()
    tabla.columns = ["Ganador", "Perdedor", "Rank G.", "Rank P.",
                     "Torneo", "Ronda", "Score", "Año"]

    with out:
        clear_output(wait=True)
        print(f"\n⚡ UPSETS (gap ≥ {gap}) ({años[0]}–{años[1]}) "
              f"— {len(upsets):,} encontrados | Top 30\n")
        print(tabla.to_string(index=False))


def pagina1_tablas(df_raw: pd.DataFrame) -> None:
    """Página 1C — Finales Grand Slam y tabla de upsets."""
    año_min = int(df_raw["year"].min())
    año_max = int(df_raw["year"].max())

    w_años_gs  = widgets.IntRangeSlider(
        value=[año_min, año_max], min=año_min, max=año_max, step=1,
        description="Años GS:", continuous_update=False,
        layout=widgets.Layout(width="480px"),
    )
    w_años_up  = widgets.IntRangeSlider(
        value=[2000, año_max], min=año_min, max=año_max, step=1,
        description="Años Upset:", continuous_update=False,
        layout=widgets.Layout(width="480px"),
    )
    w_gap = widgets.IntSlider(
        value=20, min=10, max=150, step=5,
        description="Gap mín.:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="350px"),
    )

    out_gs     = widgets.Output()
    out_upsets = widgets.Output()

    w_años_gs.observe(lambda _: _render_gs(df_raw, w_años_gs.value, out_gs), names="value")
    w_años_up.observe(lambda _: _render_upsets(df_raw, w_años_up.value, w_gap.value, out_upsets), names="value")
    w_gap.observe(lambda _: _render_upsets(df_raw, w_años_up.value, w_gap.value, out_upsets), names="value")

    print("🏆 Finales de Grand Slam")
    display(w_años_gs, out_gs)
    print("\n⚡ Upsets")
    display(widgets.HBox([w_años_up, w_gap]), out_upsets)

    _render_gs(df_raw, w_años_gs.value, out_gs)
    _render_upsets(df_raw, w_años_up.value, w_gap.value, out_upsets)


# ==============================================================================
# 5. PÁGINA 2 — Head to Head
# ==============================================================================

def _stats_golpes(df_j: pd.DataFrame, jugador: str) -> dict:
    """
    Calcula estadísticas de golpes promedio por partido para un jugador dado,
    combinando filas donde fue ganador y donde fue perdedor.
    """
    as_winner = df_j[df_j["winner_name"] == jugador]
    as_loser  = df_j[df_j["loser_name"]  == jugador]

    def safe_mean(series):
        return round(series.mean(), 1) if series.notna().any() else None

    aces      = pd.concat([as_winner["w_ace"],    as_loser["l_ace"]])
    dfs       = pd.concat([as_winner["w_df"],     as_loser["l_df"]])
    svpt      = pd.concat([as_winner["w_svpt"],   as_loser["l_svpt"]])
    first_in  = pd.concat([as_winner["w_1stIn"],  as_loser["l_1stIn"]])
    first_won = pd.concat([as_winner["w_1stWon"], as_loser["l_1stWon"]])
    second_w  = pd.concat([as_winner["w_2ndWon"], as_loser["l_2ndWon"]])
    bp_saved  = pd.concat([as_winner["w_bpSaved"],as_loser["l_bpSaved"]])
    bp_faced  = pd.concat([as_winner["w_bpFaced"],as_loser["l_bpFaced"]])

    pct_1st = (first_in / svpt * 100).mean()
    pct_1st_won = (first_won / first_in * 100).mean()
    pct_bp = (bp_saved / bp_faced * 100).mean()

    return {
        "Aces/partido":         safe_mean(aces),
        "DFs/partido":          safe_mean(dfs),
        "% 1er servicio":       round(pct_1st, 1) if pd.notna(pct_1st) else None,
        "% ganados en 1er sv.": round(pct_1st_won, 1) if pd.notna(pct_1st_won) else None,
        "% 2do servicio ganado":round((second_w / (svpt - first_in) * 100).mean(), 1) if pd.notna((second_w / (svpt - first_in) * 100).mean()) else None,
        "% BPs salvados":       round(pct_bp, 1) if pd.notna(pct_bp) else None,
    }


def _render_h2h(df_raw: pd.DataFrame, j1: str, j2: str,
                out: widgets.Output) -> None:
    mask = (
        ((df_raw["winner_name"] == j1) & (df_raw["loser_name"] == j2)) |
        ((df_raw["winner_name"] == j2) & (df_raw["loser_name"] == j1))
    )
    df_h2h = df_raw[mask].copy()

    if len(df_h2h) == 0:
        with out:
            clear_output(wait=True)
            print(f"⚠️ No hay partidos registrados entre {j1} y {j2}.")
        return

    df_h2h["resultado"] = df_h2h["winner_name"].apply(lambda w: "W" if w == j1 else "L")
    wins_j1   = (df_h2h["resultado"] == "W").sum()
    wins_j2   = (df_h2h["resultado"] == "L").sum()
    total_h2h = len(df_h2h)

    with out:
        clear_output(wait=True)

        # --- Marcador global ---
        fig_s, ax_s = plt.subplots(figsize=(10, 1.8))
        ax_s.axis("off")
        pct = wins_j1 / total_h2h
        ax_s.barh(0, pct, color="#1565C0", height=0.4)
        ax_s.barh(0, 1 - pct, left=pct, color="#C62828", height=0.4)
        ax_s.text(pct / 2, 0, f"{j1}\n{wins_j1} victorias",
                  ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax_s.text(pct + (1 - pct) / 2, 0, f"{j2}\n{wins_j2} victorias",
                  ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax_s.set_title(f"H2H — {j1} vs {j2} ({total_h2h} partidos)",
                       fontsize=13, fontweight="bold")
        ax_s.set_xlim(0, 1)
        plt.tight_layout()
        plt.show()

        # --- Desglose por superficie ---
        sup_stats = []
        for sup in SUPERFICIES_MAIN:
            df_sup = df_h2h[df_h2h["surface"] == sup]
            w1 = (df_sup["resultado"] == "W").sum()
            w2 = (df_sup["resultado"] == "L").sum()
            sup_stats.append((sup, w1, w2, len(df_sup)))

        fig_sup, axes_sup = plt.subplots(1, 3, figsize=(12, 2.5))
        for ax, (sup, w1, w2, tot) in zip(axes_sup, sup_stats):
            color = COLORES_SUPERFICIE[sup]
            if tot > 0:
                ax.barh(0, w1 / tot, color="#1565C0", height=0.4)
                ax.barh(0, w2 / tot, left=w1 / tot, color="#C62828", height=0.4)
                ax.text(0.5, -0.65, f"{w1}–{w2} ({tot} partidos)",
                        ha="center", transform=ax.transAxes, fontsize=11,
                        fontweight="bold", color=color)
            else:
                ax.text(0.5, 0.5, "Sin partidos", ha="center", va="center",
                        transform=ax.transAxes, color="#999")
            ax.set_title(sup, fontsize=11, color=color, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.axis("off")
        plt.suptitle("Desglose por superficie", fontsize=11, y=1.02)
        plt.tight_layout()
        plt.show()

        # --- Barras agrupadas ---
        sup_con_datos = [(s, w1, w2) for s, w1, w2, t in sup_stats if t > 0]
        if sup_con_datos:
            labels = [s[0] for s in sup_con_datos]
            v1 = [s[1] for s in sup_con_datos]
            v2 = [s[2] for s in sup_con_datos]
            x  = np.arange(len(labels))

            fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
            ax_bar.bar(x - 0.2, v1, width=0.35, label=j1, color="#1565C0")
            ax_bar.bar(x + 0.2, v2, width=0.35, label=j2, color="#C62828")
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(labels)
            ax_bar.set_ylabel("Victorias")
            ax_bar.set_title("Victorias por superficie", fontsize=11, fontweight="bold")
            ax_bar.legend()
            plt.tight_layout()
            plt.show()

        # --- Estadísticas de golpes ---
        stats_j1 = _stats_golpes(df_h2h, j1)
        stats_j2 = _stats_golpes(df_h2h, j2)

        cobertura = df_h2h["w_ace"].notna().mean() * 100
        print(f"\n📊 Estadísticas de golpes (promedios por partido en este H2H) "
              f"— cobertura de datos: {cobertura:.0f}%\n")

        col_w = max(len(j1), len(j2), 20)
        header = f"{'Métrica':<30} {j1:>{col_w}} {j2:>{col_w}}"
        print(header)
        print("-" * len(header))
        for metrica in stats_j1:
            v1 = stats_j1[metrica]
            v2 = stats_j2[metrica]
            v1_str = f"{v1}" if v1 is not None else "N/D"
            v2_str = f"{v2}" if v2 is not None else "N/D"
            print(f"{metrica:<30} {v1_str:>{col_w}} {v2_str:>{col_w}}")

        # --- Historial completo ---
        historial = df_h2h[[
            "resultado", "tourney_name", "surface", "round", "score", "year"
        ]].copy()
        historial.columns = ["Res. (j1)", "Torneo", "Superficie", "Ronda", "Score", "Año"]
        historial = historial.sort_values("Año", ascending=False).reset_index(drop=True)
        print(f"\n📋 Historial completo ({total_h2h} partidos):\n")
        print(historial.to_string(index=False))


def pagina2_h2h(df_raw: pd.DataFrame) -> None:
    """
    Página 2 — Head to Head interactivo.
    Al seleccionar el jugador 1, el dropdown del jugador 2
    se actualiza mostrando solo sus rivales reales.
    """
    # Mapa: jugador -> set de rivales
    rivales_de = {}
    for _, row in df_raw[["winner_name", "loser_name"]].dropna().iterrows():
        w, l = row["winner_name"], row["loser_name"]
        rivales_de.setdefault(w, set()).add(l)
        rivales_de.setdefault(l, set()).add(w)

    todos = sorted(rivales_de.keys())

    w_j1 = widgets.Dropdown(
        options=todos,
        value=todos[0],
        description="Jugador 1:",
        layout=widgets.Layout(width="300px"),
    )
    w_j2 = widgets.Dropdown(
        options=sorted(rivales_de.get(todos[0], [])),
        description="Jugador 2:",
        layout=widgets.Layout(width="300px"),
    )
    btn  = widgets.Button(description="Ver H2H", button_style="primary")
    out  = widgets.Output()

    def on_j1_change(change):
        """Actualiza las opciones de j2 al cambiar j1."""
        nuevos_rivales = sorted(rivales_de.get(change["new"], []))
        w_j2.options = nuevos_rivales
        if nuevos_rivales:
            w_j2.value = nuevos_rivales[0]

    def on_btn(_):
        _render_h2h(df_raw, w_j1.value, w_j2.value, out)

    w_j1.observe(on_j1_change, names="value")
    btn.on_click(on_btn)

    display(widgets.HBox([w_j1, w_j2, btn]), out)


# ==============================================================================
# 6. PÁGINA 3A — Tarjetas + Barras apiladas
# ==============================================================================

def _render_p3a(df_raw: pd.DataFrame, años: tuple, out: widgets.Output) -> None:
    df    = _filtrar(df_raw, años, "Todas")
    total = len(df)

    with out:
        clear_output(wait=True)

        # Tarjetas
        fig_c, axes_c = plt.subplots(1, 3, figsize=(12, 2.5))
        for ax, sup in zip(axes_c, SUPERFICIES_MAIN):
            n     = len(df[df["surface"] == sup])
            pct   = n / total * 100 if total > 0 else 0
            color = COLORES_SUPERFICIE[sup]
            ax.set_facecolor(color + "22")
            ax.text(0.5, 0.65, f"{n:,}", ha="center", va="center",
                    fontsize=22, fontweight="bold", color=color, transform=ax.transAxes)
            ax.text(0.5, 0.28, f"{pct:.1f}% del total", ha="center", va="center",
                    fontsize=10, color="#444", transform=ax.transAxes)
            ax.set_title(sup, fontsize=13, fontweight="bold", color=color)
            ax.axis("off")
        plt.suptitle("Partidos por superficie", fontsize=12, y=1.04)
        plt.tight_layout()
        plt.show()

        # Barras apiladas top 10
        top10 = df["winner_name"].value_counts().head(10).index.tolist()
        datos = {
            sup: df[df["surface"] == sup]["winner_name"].value_counts().reindex(top10, fill_value=0)
            for sup in SUPERFICIES_MAIN
        }

        fig_st, ax_st = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(top10))
        for sup in SUPERFICIES_MAIN:
            vals = datos[sup].values
            ax_st.barh(top10, vals, left=bottom,
                       color=COLORES_SUPERFICIE[sup], label=sup, edgecolor="white")
            bottom += vals
        ax_st.set_title("Top 10 jugadores — victorias por superficie (apilado)",
                        fontsize=12, fontweight="bold")
        ax_st.set_xlabel("Victorias")
        ax_st.legend(title="Superficie")
        ax_st.invert_yaxis()
        plt.tight_layout()
        plt.show()


def pagina3_superficie_overview(df_raw: pd.DataFrame) -> None:
    """Página 3A — Tarjetas por superficie y barras apiladas top 10."""
    año_min = int(df_raw["year"].min())
    año_max = int(df_raw["year"].max())

    w_años = widgets.IntRangeSlider(
        value=[año_min, año_max], min=año_min, max=año_max, step=1,
        description="Años:", continuous_update=False,
        layout=widgets.Layout(width="500px"),
    )
    out = widgets.Output()

    w_años.observe(lambda _: _render_p3a(df_raw, w_años.value, out), names="value")
    display(w_años, out)
    _render_p3a(df_raw, w_años.value, out)


# ==============================================================================
# 7. PÁGINA 3B — Comparativa de métricas + Ranking por superficie
# ==============================================================================

def _render_p3b(df_raw: pd.DataFrame, años: tuple, sup_ranking: str,
                out: widgets.Output) -> None:
    df = _filtrar(df_raw, años, "Todas")

    with out:
        clear_output(wait=True)

        # Comparativa de métricas
        metricas = []
        for sup in SUPERFICIES_MAIN:
            df_s = df[df["surface"] == sup]
            pct_1st = np.nanmean([
                (df_s["w_1stIn"] / df_s["w_svpt"] * 100).mean(),
                (df_s["l_1stIn"] / df_s["l_svpt"] * 100).mean(),
            ])
            metricas.append({
                "Superficie":          sup,
                "% 1er servicio":      round(pct_1st, 1),
                "Duración media (min)":round(df_s["minutes"].mean(), 1),
                "Aces por partido":    round((df_s["w_ace"].fillna(0) + df_s["l_ace"].fillna(0)).mean(), 1),
            })
        df_met = pd.DataFrame(metricas).set_index("Superficie")

        fig_m, axes_m = plt.subplots(1, 3, figsize=(13, 4))
        for ax, col in zip(axes_m, df_met.columns):
            colores = [COLORES_SUPERFICIE[s] for s in df_met.index]
            ax.bar(df_met.index, df_met[col], color=colores, edgecolor="white", width=0.5)
            ax.set_title(col, fontsize=10, fontweight="bold")
            for i, v in enumerate(df_met[col]):
                ax.text(i, v + max(df_met[col]) * 0.01, str(v), ha="center",
                        fontsize=10, fontweight="bold")
        plt.suptitle(f"Comparativa de métricas por superficie ({años[0]}–{años[1]})",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

        cob_min = df[df["surface"].isin(SUPERFICIES_MAIN)]["minutes"].notna().mean() * 100
        cob_ace = df[df["surface"].isin(SUPERFICIES_MAIN)]["w_ace"].notna().mean() * 100
        print(f"  ℹ️  Cobertura — Duración: {cob_min:.0f}% | Aces: {cob_ace:.0f}%")

        # Ranking por superficie
        df_sr  = df[df["surface"] == sup_ranking].copy()
        victorias = df_sr["winner_name"].value_counts().rename("Victorias")
        partidos  = (
            df_sr["winner_name"].value_counts()
            .add(df_sr["loser_name"].value_counts(), fill_value=0)
        )
        winrate = (victorias / partidos * 100).round(1)

        aces_w  = df_sr.groupby("winner_name")["w_ace"].sum()
        aces_l  = df_sr.groupby("loser_name")["l_ace"].sum()
        aces_m  = (aces_w.add(aces_l, fill_value=0) / partidos).round(1)

        ranking = pd.DataFrame({
            "Victorias":    victorias,
            "Partidos":     partidos,
            "Winrate (%)":  winrate,
            "Aces/partido": aces_m,
        }).dropna(subset=["Victorias"])
        ranking = ranking[ranking["Partidos"] >= 20].sort_values("Victorias", ascending=False).head(25)
        ranking.index.name = "Jugador"

        print(f"\n🏅 Ranking en {sup_ranking} (mín. 20 partidos) — top 25:\n")
        print(ranking.to_string())


def pagina3_metricas_ranking(df_raw: pd.DataFrame) -> None:
    """Página 3B — Comparativa de métricas por superficie y ranking filtrable."""
    año_min = int(df_raw["year"].min())
    año_max = int(df_raw["year"].max())

    w_años = widgets.IntRangeSlider(
        value=[2000, año_max], min=año_min, max=año_max, step=1,
        description="Años:", continuous_update=False,
        layout=widgets.Layout(width="500px"),
    )
    w_sup = widgets.Dropdown(
        options=SUPERFICIES_MAIN, value="Hard",
        description="Superficie ranking:",
        style={"description_width": "initial"},
    )
    out = widgets.Output()

    def on_change(_):
        _render_p3b(df_raw, w_años.value, w_sup.value, out)

    w_años.observe(on_change, names="value")
    w_sup.observe(on_change, names="value")

    display(widgets.HBox([w_años, w_sup]), out)
    _render_p3b(df_raw, w_años.value, w_sup.value, out)
