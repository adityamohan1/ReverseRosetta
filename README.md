# ReverseRosetta

ReverseRosetta is a production-oriented Python package that turns **protein sequences** from Excel into **human codon–optimized DNA** coding sequences. It keeps the amino acid translation fixed, enforces a clean ORF, removes a configurable set of **restriction enzyme sites**, and applies soft objectives for **cryptic splice–like motifs** (via SpliceFinder-compatible scoring or a fast heuristic) and **synthesis/repeat burden**.

## Features

- **Stage 1 — CodonTransformer**: initial DNA from the published BigBird model (`adibvafa/CodonTransformer`) with `match_protein=True` so decoding stays within synonymous codons per residue.
- **Stage 2 — Restriction removal**: iterative synonymous edits ranked by codon preference, splice/repeat impact, and hit count.
- **Stage 3 — Splice minimization**: local synonymous search around high-scoring donor/acceptor windows.
- **Stage 4 — Repeat minimization**: reduce homopolymer/tandem/k-mer/complexity penalties while preserving hard constraints.
- **Stage 5 — Validation**: translation, ORF, alphabet, internal stops, and residual splice warnings.

Configuration is merged from **CLI flags** and an optional **YAML** file (`pydantic` models).

## Installation

Requires **Python 3.11+** (tested on 3.12 as well).

```bash
cd /path/to/ReverseRosetta
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Core dependencies include `CodonTransformer`, `torch`, `transformers`, `pandas`, `openpyxl`, `pydantic`, and `typer`. The first CodonTransformer run downloads Hugging Face weights.

### Optional: SpliceFinder Keras models

The upstream SpliceFinder release targets **TensorFlow 1 / Keras 2**. This repo includes a **TensorFlow 2 `tf.keras`** loader for `CNN.h5` (and optional `donor_dis.h5` / `acceptor_dis.h5`) when you enable Keras mode. Install TensorFlow separately:

```bash
pip install 'tensorflow>=2.14,<3'
```

If models are missing or loading fails, the pipeline **falls back to a deterministic heuristic** splice scorer (GT/AG context–based) so installs stay usable without TF.

## Quick start

Generate or refresh the example workbook and print a sample command:

```bash
PYTHONPATH=. python scripts/run_example.py
```

Run the CLI (downloads models on first use):

```bash
python -m reverserosetta \
  --input examples/ReverseRosettaTemplate.xlsx \
  --sheet Sheet1 \
  --output results/reverserosetta_output.xlsx \
  --host human
```

Or after `pip install -e .`:

```bash
reverserosetta --input examples/ReverseRosettaTemplate.xlsx --sheet Sheet1 --output results/out.xlsx
```

### Useful flags

| Flag | Purpose |
|------|---------|
| `--column-index` | 1-based Excel column for amino acids (default **4** = column D). |
| `--emit-stop-codon` | Append a terminal stop (`TAA`/`TAG`/`TGA` chosen to avoid restriction motifs when possible). |
| `--max-iterations` | Global budget for refinement loops. |
| `--splice-donor-threshold` / `--splice-acceptor-threshold` | Flag significant splice signals (scale depends on backend). |
| `--config` | YAML overrides (see `examples/config.example.yaml`). |
| `--output-dir` | Where to place the companion CSV (and optional JSON audits). |
| `--emit-json-reports` | Write per-sequence JSON under `<output-dir>/audits/`. |
| `--use-splicefinder-keras` + `--splicefinder-model-dir` | Use downloaded `CNN.h5`. |

## Input Excel format

- **Sheet**: name passed with `--sheet`.
- **Column**: `--column-index` (default 4) is the **1-based** column index of **one-letter amino acid sequences**.
- Blank cells are skipped; strings are stripped and uppercased.
- Only standard amino acid letters are accepted (see `reverserosetta.utils.VALID_PROTEIN`). Invalid characters **fail loudly**.

Trailing empty columns are padded internally so “column D” works even if Excel omitted blank trailing columns.

## Output

- **Excel** (`.xlsx`) and **CSV** with: row index, Excel row, original AA, optimized DNA, lengths, restriction cleanup flag, splice metrics before/after, repeat scores before/after, validation status, errors/warnings.
- **Console**: the same table is printed.
- **Optional JSON** per sequence: initial/final DNA, edit log, splice/repeat summaries.

## Project layout

```
reverserosetta/
  cli.py          # Typer CLI
  config.py       # Pydantic settings + YAML merge
  excel_io.py     # Excel ingestion
  codon_opt.py    # CodonTransformer adapter (lazy imports)
  restriction.py  # Motifs + synonymous proposals
  splice.py       # SpliceFinder Keras + heuristic backend
  repeats.py      # Manufacturability scoring
  validate.py     # Hard checks
  optimize.py     # Multi-stage orchestration
  reporting.py    # Tables + exports
  utils.py        # Genetic code, human codon weights, logging
tests/            # pytest suite (smoke tests avoid downloading HF weights)
examples/         # Template workbook + sample YAML
scripts/run_example.py
```

## How CodonTransformer is used

`reverserosetta/codon_opt.py` calls `CodonTransformer.CodonPrediction.predict_dna_sequence` with:

- `organism="Homo sapiens"` when `--host human`
- `match_protein=True`
- `deterministic=True`
- Hugging Face weights `adibvafa/CodonTransformer`

The returned `predicted_dna` string is sanitized to uppercase A/C/G/T.

## How SpliceFinder is used

- **Heuristic mode (default)**: scans for donor/acceptor-like `GT` / `AG` dinucleotides with simple context scores in `[0, 1]` for thresholding.
- **Keras mode**: loads `CNN.h5` from `--splicefinder-model-dir`, runs sliding 400 nt windows (padded with `A`), and optionally applies `donor_dis.h5` / `acceptor_dis.h5` filtering like the upstream `test_Cla.py` sketch. Class indices follow the original three-way head (acceptor / donor / non-splice).

Thresholds are **backend-dependent**; tune `--splice-donor-threshold` / `--splice-acceptor-threshold` when switching backends.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

Smoke tests inject a deterministic codon chooser so CI does not require Hugging Face downloads.

## Known limitations

- **Restriction semantics**: motifs use published core sequences (and degenerate regex where noted). Type IIS enzymes use the **recognition core** only; real manufacturing may need vendor-specific spacing rules.
- **BseRI**: approximated as `CC[AT]GG` (NEB `CCWGG`).
- **SpliceFinder**: original code targets old TensorFlow; TF2 loading may require matching protobuf/`h5` versions. The heuristic backend is always available.
- **Optimization**: greedy local search with an iteration budget; difficult sequences may retain restriction hits or high splice scores—check `validation_ok` and warnings.
- **BigBird / long proteins**: CodonTransformer memory scales with length; very long inputs may need chunking (not implemented here).

## Future extensions

- Pluggable codon engines (CAI tables only, other organisms).
- MILP / simulated annealing for global constraint satisfaction.
- Chunked CodonTransformer inference for long ORFs.
- Tighter integration with MaxEntScan or official SpliceFinder TF2 ports.

## GitHub & documentation website

This repo includes a small static homepage in **`docs/index.html`** for [GitHub Pages](https://pages.github.com/).

1. Create a **new repository** on GitHub (e.g. `ReverseRosetta`), **without** initializing a README (you already have one locally).
2. On your machine, from the project root:

   ```bash
   git init
   git add .
   git commit -m "Initial commit: ReverseRosetta pipeline"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ReverseRosetta.git
   git push -u origin main
   ```

3. In **`docs/index.html`**, set `USER` and `REPO` in the `<script>` at the bottom (replace `YOUR_GITHUB_USERNAME` and adjust `REPO` if your repo name differs), commit, and push.
4. On GitHub: **Settings → Pages → Build and deployment → Source**: choose **Deploy from a branch**, branch **`main`**, folder **`/docs`**, Save.
5. After a minute, the site will be at **`https://YOUR_USERNAME.github.io/ReverseRosetta/`** (URL uses your **username** and **repository name**).

Optional: set the repo **Website** field (**Settings → General → Website**) to that Pages URL so it shows on the repository front page.

## License

MIT (project scaffold). Third-party models (CodonTransformer, SpliceFinder) follow their respective licenses.
