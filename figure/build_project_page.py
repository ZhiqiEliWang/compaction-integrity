"""Render only the original project-page results from the published arXiv v2 source.

Requires matplotlib and pymupdf. Run from the repository root:
    python figure/build_project_page.py
"""

import argparse
import io
from pathlib import Path
import re
import tarfile
from urllib.request import urlopen

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pymupdf

SOURCE_URL = "https://arxiv.org/src/2608.11242v2"
OUTPUT = Path(__file__).resolve().parents[1] / "docs/static"
FIGURES = {
    "fig1-failure-mode.svg": "figures/intro_figure/fig1.pdf",
    "fig2-compint-overview.svg": "figures/benchmark/conpact.pdf",
    "fig3-retention-dot.svg": "figures/rq1_dot/dot_retention_compliance.pdf",
    "fig5-retention-by-position.svg": "figures/100k_positions/retention_rate_by_position_all__100k.pdf",
    "fig6-retention-by-type.svg": "figures/diff_sc_type/diff_sc_type_plot.pdf",
}


def build(output: Path, source_archive: Path | None):
    source = source_archive.read_bytes() if source_archive else urlopen(SOURCE_URL).read()
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(source)) as archive:
        for filename, member in FIGURES.items():
            pdf = archive.extractfile(member).read()
            with pymupdf.open(stream=pdf, filetype="pdf") as doc:
                page = doc[0]
                (output / filename).write_text(page.get_svg_image(text_as_path=True))
                if filename in {"fig1-failure-mode.svg", "fig2-compint-overview.svg"}:
                    page.get_pixmap(matrix=pymupdf.Matrix(4, 4)).save(output / filename.replace(".svg", ".png"))

        table = archive.extractfile("tables/rq4.tex").read().decode()
        rows = re.findall(r"^(Hermes Agent|OpenResearcher|WildChat)\s*&\s*[\d.]+\s*&\s*([\d.]+)", table, re.M)
        labels, values = zip(*rows)
        values = [float(value) for value in values]

    plt.rcParams.update({"font.family": "serif", "svg.hashsalt": "arxiv-2608.11242v2"})
    fig, ax = plt.subplots(figsize=(6, 1.8), layout="constrained")
    ax.scatter(values, range(len(labels)), color="#97374f", s=35, zorder=3)
    for y, value in enumerate(values):
        ax.annotate(f"{value:.1f}%", (value, y), xytext=(-9, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=11)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlim=(0, 100), ylim=(2.5, -0.5), xlabel="SC registry retention")
    ax.xaxis.set_major_locator(plt.MultipleLocator(25))
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="x", color="0.9")
    ax.tick_params(axis="both", length=0, labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    path = output / "fig7-extractor-dot.svg"
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(f"Rendered the published arXiv v2 figures and extractor retention in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, help="Downloaded arXiv v2 source archive")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.output, args.source_archive)
