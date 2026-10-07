# visualizations of the side constraints themselves (no compaction results):
# a word cloud of the hand-annotated SWE-natural SCs

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

from datasets import Dataset
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from nltk.stem import WordNetLemmatizer
import pandas as pd
from wordcloud import STOPWORDS, WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from compaction_integrity.viz_config import PALETTE, set_paper_style, save_fig
from compaction_integrity.analyze.utils import tee_stdout
from compaction_integrity.dataset.swe_natural_curation.dataset import NEGATION_PATTERN


DEFAULT_DATASET_PATH = Path(
    "/data/compaction_integrity/default_ds/swe_natural_sc_100k_n126/stitched_dataset"
)
DEFAULT_OUTPUT_DIR = Path("/data/compaction_integrity/analysis/swe_natural_sc_n126")

# Fixed over all five COMPINT types so a type keeps its color across figures,
# even when a dataset only has some of them. Every pair can touch in a word cloud,
# so these are the 5-of-7 palette colors with the best worst-pair CVD separation
# (sage and violet dropped); the weakest pair is Process/Output.
SC_TYPE_ORDER = ["Action", "Information", "Process", "Preference", "Output"]
SC_TYPE_COLORS = dict(zip(SC_TYPE_ORDER, [PALETTE[i] for i in (0, 6, 2, 1, 5)]))
MIXED_COLOR = "#8c8c8c"

# Every SC passed the deontic keyword screen, so its terms are in the text by
# construction and would say more about the screen than about the SCs.
SCREEN_WORDS = {word for phrase in NEGATION_PATTERN.split("|") for word in phrase.split()}
WORDCLOUD_STOPWORDS = {word.lower() for word in STOPWORDS} | SCREEN_WORDS

MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
BARE_URL = re.compile(r"https?://\S+")
MENTION = re.compile(r"@[\w-]+")
WORD = re.compile(r"[a-z][a-z'-]*[a-z]")

LEMMATIZER = WordNetLemmatizer()


def _lemma(word: str) -> str:
    """Fold inflections so check/checked/checking count as one word.

    One lemmatizer pass only: chaining noun then verb turns "bits" into "bite".
    WordNet's noun rule also strips the final s of "pass", hence the -ss guard.
    """
    verb = LEMMATIZER.lemmatize(word, "v")
    if verb != word or word.endswith("ss"):
        return verb
    return LEMMATIZER.lemmatize(word, "n")


def _load_sc_clauses(dataset_path: Path) -> pd.DataFrame:
    """One row per annotated clause: the exact spans removed in the no-SC arm.

    Read from the built eval dataset rather than the annotation CSVs, so the
    cloud always covers exactly the SCs that were evaluated.
    """
    dataset = Dataset.load_from_disk(str(dataset_path))
    rows = [
        {"annotation_id": annotation_id, "sc_type": sc_type, "clause_text": clause}
        for annotation_id, sc_type, clauses in zip(
            dataset["annotation_id"], dataset["sc_type"], dataset["sc_clauses"]
        )
        for clause in clauses
    ]
    return pd.DataFrame(rows)


def _tokenize(text: str) -> list[str]:
    text = MARKDOWN_LINK.sub(r"\1", text)
    text = BARE_URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = text.lower().replace("’", "'")
    words = [
        word
        for word in WORD.findall(text)
        if len(word) > 2 and word not in WORDCLOUD_STOPWORDS
    ]
    return [lemma for lemma in map(_lemma, words) if lemma not in WORDCLOUD_STOPWORDS]


def _word_frequencies(clauses_df: pd.DataFrame) -> pd.DataFrame:
    """Weight each word by the number of SCs it appears in.

    Document frequency rather than raw counts: the multi-step reproduction SCs
    repeat their own nouns ("device" five times in one SC), which would
    otherwise dominate a cloud built from only 22 SCs.
    """
    sc_words = defaultdict(list)
    sc_types = {}
    for clause in clauses_df.itertuples():
        sc_words[clause.annotation_id].extend(_tokenize(clause.clause_text))
        sc_types[clause.annotation_id] = clause.sc_type

    n_scs = Counter()
    n_occurrences = Counter()
    types_by_word = defaultdict(set)
    for annotation_id, words in sc_words.items():
        n_occurrences.update(words)
        n_scs.update(set(words))
        for word in set(words):
            types_by_word[word].add(sc_types[annotation_id])

    freq_df = pd.DataFrame(
        {
            "word": list(n_scs),
            "n_scs": [n_scs[word] for word in n_scs],
            "n_occurrences": [n_occurrences[word] for word in n_scs],
            "sc_types": [
                ";".join(t for t in SC_TYPE_ORDER if t in types_by_word[word]) for word in n_scs
            ],
        }
    )
    return freq_df.sort_values(
        ["n_scs", "n_occurrences", "word"], ascending=[False, False, True]
    ).reset_index(drop=True)


def _plot_wordcloud(freq_df: pd.DataFrame, output_path: Path, seed: int) -> None:
    """Size by number of SCs; color by SC type, gray when shared across types."""
    word_colors = {
        row.word: SC_TYPE_COLORS[row.sc_types] if ";" not in row.sc_types else MIXED_COLOR
        for row in freq_df.itertuples()
    }
    serif_font = font_manager.findfont(
        font_manager.FontProperties(family=plt.rcParams["font.serif"])
    )
    cloud = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        font_path=serif_font,
        prefer_horizontal=0.9,
        relative_scaling=0.8,
        max_font_size=220,
        min_font_size=14,
        margin=4,
        random_state=seed,
        color_func=lambda word, **_: word_colors[word],
    ).generate_from_frequencies(dict(zip(freq_df["word"], freq_df["n_scs"])))

    listed_types = set(";".join(freq_df["sc_types"]).split(";"))
    present_types = [t for t in SC_TYPE_ORDER if t in listed_types]
    handles = [Patch(color=SC_TYPE_COLORS[t], label=t) for t in present_types]
    if freq_df["sc_types"].str.contains(";").any():
        handles.append(Patch(color=MIXED_COLOR, label="several types"))
    # One row of five entries is wider than a single-column figure.
    n_legend_cols = min(len(handles), 3)
    n_legend_rows = -(-len(handles) // n_legend_cols)

    width_inch = plt.rcParams["figure.figsize"][0]
    fig, ax = plt.subplots(
        figsize=(width_inch, width_inch * 900 / 1600 + 0.2 * n_legend_rows + 0.1)
    )
    ax.imshow(cloud.to_array(), interpolation="bilinear")
    ax.set_axis_off()
    # "outside" makes constrained layout reserve room below the cloud.
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=n_legend_cols,
        frameon=False,
        handlelength=0.8,
        columnspacing=1.0,
    )
    save_fig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=Path, default=DEFAULT_DATASET_PATH)
    args.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args.add_argument("--seed", type=int, default=42)
    parsed_args = args.parse_args()

    output_dir = parsed_args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with tee_stdout(output_dir / "viz_sc.log"):
        set_paper_style(use_latex=False)

        clauses_df = _load_sc_clauses(parsed_args.dataset_path)
        freq_df = _word_frequencies(clauses_df)
        freq_df.to_csv(output_dir / "wordcloud_frequencies__swe_natural_sc.csv", index=False)

        n_scs = clauses_df["annotation_id"].nunique()
        print(
            f"{n_scs} SCs, {len(clauses_df)} clauses, {len(freq_df)} distinct words "
            f"({(freq_df['n_scs'] == 1).sum()} appear in only one SC)"
        )
        print(freq_df.head(20).to_string(index=False))

        _plot_wordcloud(freq_df, output_dir / "wordcloud__swe_natural_sc.pdf", parsed_args.seed)
