# Project-page figures

The page keeps its original result sections. Figures and values come exclusively from
[arXiv:2608.11242v2](https://arxiv.org/abs/2608.11242v2), published October 11, 2026.

| Page figure | Published source |
| --- | --- |
| 1: failure mode | Figure 1; `figures/intro_figure/fig1.pdf` |
| 2: evaluation suite | Figure 2; `figures/benchmark/conpact.pdf` |
| 3: retention dot plot | Figure 4; `figures/rq1_dot/dot_retention_compliance.pdf` |
| 4: injection location | Figure 7; `figures/100k_positions/retention_rate_by_position_all__100k.pdf` |
| 5: SC type | Figure 8; `figures/diff_sc_type/diff_sc_type_plot.pdf` |
| 6: extractor retention dot plot | Table 4, extractor-registry retention: Hermes Agent 93.7%, OpenResearcher 95.1%, WildChat 90.3% |

The first five figures are converted directly from the PDFs in the published source
archive. The extractor dot plot replaces the original extractor table and reads only
its registry-retention column from published Table 4. The two latency values in the page
text are reported in Section 5.4. The former Hermes Agent latency value is omitted
because it is not reported in arXiv v2.

The page uses the paper's 17% overall retention headline. Open-weight compactors and
baselines use 100K-token contexts (N = 750); GPT-5.4-mini uses 220K-token contexts
(N = 300). These conditions remain distinguished in the dot-plot caption.

Rebuild from the repository root with `matplotlib` and `pymupdf` installed:

```bash
python figure/build_project_page.py
```

The builder downloads the version-pinned source archive from
`https://arxiv.org/src/2608.11242v2`. To reuse a downloaded archive:

```bash
python figure/build_project_page.py --source-archive /path/to/arxiv-v2-source.tar.gz
```

It does not read current experiment outputs or run models. Replacing published figures
with newer experiment results requires a corresponding paper update.
