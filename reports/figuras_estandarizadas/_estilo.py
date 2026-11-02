"""Estilo visual compartido — todas las figuras del proyecto en un solo color.

Color base: AZUL profesional #4c72b0 (azul seaborn).
Todas las figuras del pipeline (01→06) se regeneran importando este módulo, para
que compartan exactamente la misma paleta, tipografía y estética.

Uso típico:

    import sys; sys.path.insert(0, "reports/figuras_estandarizadas")
    from _estilo import AZUL, aplicar_estilo, paleta, CMAP, CMAP_DIV, guardar

    aplicar_estilo()
    fig, ax = plt.subplots()
    ax.bar(x, h, color=AZUL)                 # 1 serie -> color base
    for c, col in zip(cols, paleta(len(cols))):  # varias series -> degradado azul
        ax.plot(..., color=col)
    sns.heatmap(M, cmap=CMAP)                 # heatmap -> azul secuencial
    guardar(fig, "05_baseline__01_auc")      # -> reports/figuras_estandarizadas/<nombre>.png
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Color base y familia ---------------------------------------------------
AZUL = "#4c72b0"          # primario
AZUL_OSCURO = "#2c4a80"   # énfasis / contraste dentro de la misma familia
AZUL_CLARO = "#c6d3e8"    # relleno suave / fondo de banda
GRIS = "#8a8a8a"          # elementos neutros (líneas de referencia, texto auxiliar)

# Carpeta destino (este archivo vive dentro de ella)
DIR_FIG = Path(__file__).resolve().parent

# Rampa monocroma para series categóricas / múltiples: claro -> base -> oscuro
_CMAP = LinearSegmentedColormap.from_list(
    "azul_glamour", ["#eaf0f8", AZUL_CLARO, AZUL, AZUL_OSCURO, "#1b2f52"])
# Alias públicos
CMAP = _CMAP                     # secuencial azul (heatmaps, SHAP, intensidades)
CMAP_R = _CMAP.reversed()
# Diverging construido sobre la MISMA familia (gris neutro al centro), por si
# alguna figura necesita separar signo sin salir del azul.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "azul_div", [AZUL_OSCURO, AZUL, "#f2f2f2", "#e6a0a0", "#b0453f"])


def paleta(n):
    """n colores del degradado azul (para series múltiples / categorías).

    Con n=1 devuelve el color base. Con n>=2 muestrea la rampa evitando los
    extremos demasiado claros para que todo se lea con buen contraste.
    """
    if n <= 1:
        return [AZUL]
    lo, hi = 0.30, 0.95
    return [_CMAP(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def aplicar_estilo():
    """Fija rcParams globales: rejilla limpia, spines discretos, azul por defecto."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.size": 10.5,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        # Ciclo de color por defecto = familia azul (para código que no pasa color=)
        "axes.prop_cycle": mpl.cycler(color=[AZUL, AZUL_OSCURO, AZUL_CLARO, GRIS]),
        "image.cmap": "Blues",
    })


def guardar(fig, nombre):
    """Guarda la figura como PNG en la carpeta estándar y la cierra.

    `nombre` sin extensión. Devuelve la ruta escrita.
    """
    if not nombre.endswith(".png"):
        nombre += ".png"
    ruta = DIR_FIG / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta
