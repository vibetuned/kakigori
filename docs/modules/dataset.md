# Datasets — built with musicset

kakigori no longer builds its own data. Selection, engraving, annotation,
groundtruth graphs, `**kern`, scan-like augmentation and synthetic scores come
from [musicset](../../../musicset) (a dependency of this package, so
`uv run musicset ...` works from this repo). The conf directories (`conf/`,
`conf-lite/`, `conf-lite-pruned/`...) are musicset label sets as they are:
`--labels conf-lite`.

## Layout

Each `data/<set>/` keeps its curated sources in `filter/` (`.mxl`); a build
writes next to them:

| Directory | Content | Read by |
| --- | --- | --- |
| `imgs/<score>_page<N>.png` | rasterized pages | vision + graph training |
| `annotations/<score>_page<N>.json` | detection boxes (`class`, `bbox`, `id`, `parent`) | vision + graph training |
| `graphs/<score>.json` | groundtruth graph (`node_ids`, `edge_index`, `y`, `json_files`) | graph training, `validate-*` |
| `mei/<score>.mei` | MEI groundtruth (ids shared with the pages) | `compare-kern`, `validate-*` |
| `kern/<score>.krn` | `**kern` read off the annotations and graph | reference |
| `pages/`, `manifest/`, `injected/`, `svgs/` | canonical annotations, build settings, sources with stable ids, SVG pages | musicset (re-projection, viewer) |

Graphs are JSON; `kakigori.graph.graph_io` loads them (and the `.pt` graphs of
datasets built by the old pipeline) as the same dict of tensors.

## Building

```bash
./regenerate-dataset.sh data/train-lite                         # varied engraving, seed 0
SEED=1 ./regenerate-dataset.sh data/validation-lite
PROFILE=fixed CONF=conf ./regenerate-dataset.sh data/train-small
```

which runs `musicset build data/<set>/filter data/<set> --labels $CONF --graph
--kern --profile $PROFILE --seed $SEED`. Builds are resumable and never stale:
each score's manifest records its settings, a score is skipped only when they
match, and a changed label set re-projects `annotations/` and `graphs/` from
`pages/` and `mei/` without rendering. Rejected scores (pages with a staff off
the page, or the optional density gates) are listed in `failures.jsonl`.

`--profile varied` gives every score its own font (Leipzig, Bravura,
Gootville, Leland, Petaluma), staff size, line and stem weights, spacing and
page; `--variants K` engraves each score K times. `fixed` is Verovio's default
look.

## Choosing sources and making new ones

```bash
# lite sources: one part, <= 2 staves, one voice per staff, no tab/percussion
uv run musicset select ../music_data/mxl data/train-lite/filter --lite --limit 9500 \
    --exclude-from data/validation-lite/filter --copy

# synthetic scores, then build them like any other set
uv run musicset synth exercise data/synthetic-lite/filter -n 1500 --profile detector
uv run musicset synth mutate data/synthetic-lite/filter --sources data/train-lite/filter -n 1500
```

`musicset synth` also writes piano pieces over functional harmony and species
counterpoint (duets with C clefs, choir with lyrics); see musicset's README.

## Augmentation and inspection

```bash
uv run musicset augment data/train-lite data/train-lite-scans --preset mixed --copies 2
uv run musicset view data/train-lite --labels conf-lite --scans test/
```

`musicset.augment.Augmenter` can also run online in a training dataset (on the
full-resolution page, before the 640 px letterbox: thin strokes are only a
pixel wide there).

## Old commands

| Old (removed) | Now |
| --- | --- |
| `filter-mxl --lite` | `musicset select --lite` |
| `inject-mids`, `render-dataset`, `extract-annotations`, `export-dataset`, `generate-graphs` | `musicset build` (one step; `--graph --kern`) |
| `synthetic-writer`, `synthetic-generator`, `synthetic-arranger` | `musicset synth exercise` / `mutate` / `arrange` |
| `visualize-dataset`, `visualize-graphs`, `visualize-graphs-pysigma` | `musicset view` |
| `extract-classes`, `frequencies` | label sets are fixed confs; class counts: `musicset view` class filter or a few lines over `annotations/` |
