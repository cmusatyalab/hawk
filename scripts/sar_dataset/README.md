# Search-and-rescue mission data for Hawk (C2A + AIDER)

`build_sar_mission.py` turns two public aerial datasets into one Hawk mission:
128 px tiles with exactly one label each, a shuffled stream file, a bootstrap
zip and a config snippet. Only Python 3.8+ and Pillow are needed.

## Reproduce our data in one command

```bash
python scripts/sar_dataset/build_sar_mission.py all --dest ~/sar_data
```

This downloads both datasets (about 5.2 GB), checks their MD5 sums, extracts
only the parts we use (about 6 GB), builds `~/sar_data/mission_v1` (about
1 GB, 164,706 tiles) and validates it. Each step can also be run on its own:

| Step | What it does |
|---|---|
| `download --dest DIR` | Fetches `AIDER.zip` (Zenodo) and `C2A_Dataset.zip` (Google Drive) and checks their MD5 sums. Skips files that are already there. |
| `extract --dest DIR` | Extracts AIDER's `normal` folder and C2A's images and pose labels into `DIR/raw`. Skips files that are already there. |
| `build --c2a-root ... --aider-root ... --out ...` | Tiles, labels and merges the data (details below). `--dry-run` writes only `stats.json`. |
| `validate --out ...` | Checks that every stream tile exists, opens and is 128×128, that the class counts match `stats.json`, and that no bootstrap image is in the stream. `--write-manifest` records a SHA-256 for every file; without it, the files are checked against that manifest. |

A long build can be split up: with `--max-seconds N` it stops after N seconds,
and the same command continues where it left off.

### Check that you got the same data

With the same inputs and the default settings (seed 0), you should get exactly:

| | Tiles |
|---|---|
| Stream, total | 164,506 |
| Negative | 156,281 |
| Upright | 3,329 |
| Sitting | 2,050 |
| Kneeling | 1,229 |
| Bent | 1,200 |
| Lying | 417 |
| Bootstrap | 20 per pose + 100 negatives |

and `index/stream.txt` with SHA-256
`6ca8a2f7baadd85210480916e6bf75d3089c70e4c48c378ab00758d44cd4b52b`
(`validate` prints it).

## The source datasets

| Dataset | What it is | Source | MD5 of the zip we used | License |
|---|---|---|---|---|
| AIDER | Real aerial photos of fires, floods, collapsed buildings, traffic accidents and normal scenes. No labeled people. | Kyrkou and Theocharides, [Zenodo record 3888300](https://zenodo.org/records/3888300) | `1ad4eb02ed156e8dfa19986ff382e58b` | CC BY 4.0 |
| C2A | 10,215 AIDER disaster photos with synthetic people pasted in, about 35 per image, each with a box and one of five poses (bent, kneeling, lying, sitting, upright). | Nihal et al., [arXiv 2408.04922](https://arxiv.org/abs/2408.04922), data at [github.com/Ragib-Amin-Nihal/C2A](https://github.com/Ragib-Amin-Nihal/C2A) (Google Drive, or [Kaggle](https://www.kaggle.com/datasets/rgbnihal/c2a-dataset)) | `d836a206fb7c682c893fd47665ea24f1` | Not stated |

C2A's repository states no license, so we share the recipe (this script), not
the tiles. If the Google Drive download fails (Drive sometimes asks for a
browser), download `C2A_Dataset.zip` by hand into the `--dest` folder and run
the command again. The Kaggle zip may differ byte for byte: add `--no-verify`.

## How the data is built

**Tiles, not boxes.** Hawk's `dnn_classifier` gives one class per tile. Boxes
with real coordinates are all stored as class 1 by the scout's data manager,
so the five poses would collapse into one class if C2A were fed in with its
boxes. The script cuts every image into a grid of 128 px tiles and labels each
tile:

| Tile contents | Label |
|---|---|
| No person | negative (0) |
| People of one pose, at least half of each box inside (`--min-visible`) | that pose (1 to 5) |
| People of different poses | dropped, or the majority pose with `--mixed majority` |
| Only cut-off people | dropped (we can't tell what it shows) |

Labels: 0 negative, 1 bent, 2 kneeling, 3 lying, 4 sitting, 5 upright, the
order of `dataset.class_list` in the snippet.

**Why 128 px.** C2A packs about 35 people of mixed poses into each image:

| Tile size | Single-pose tiles | Mixed-pose tiles (dropped) | Empty tiles |
|---|---|---|---|
| 256 px | 7,886 | 52,300 | 13,422 |
| 128 px | 54,281 | 93,237 | 92,978 |

**Why mix in AIDER.** In C2A alone nearly every image is full of people, so
nothing is rare. Hawk is built for rare targets. We add AIDER's `normal`
scenes as negatives until people are 5% of the stream (`--positive-rate`). We
use only the `normal` folder because C2A pasted its people onto AIDER's
disaster photos, and adding those photos back without people could teach the
model to spot the pasting instead of the person. Empty C2A tiles are also used
as negatives (`--no-c2a-empty-negatives` turns that off).

**Why one pose is rare.** C2A's poses are roughly balanced (about 10,000 to
13,000 single-pose tiles each). We choose the mix among the people:
40% upright, 25% sitting, 15% bent, 15% kneeling, 5% lying (`--pose-mix`;
`--pose-mix natural` keeps C2A's own mix). This is a designed condition for
the experiment, not a measurement of real disasters.

**Bootstrap.** 20 tiles per pose from C2A's train split and 100 AIDER tiles.
Every tile from a bootstrap source image is kept out of the stream.

## Use it in a Hawk mission

Copy the output folder to each scout's data root, paste `mission_snippet.yml`
into the mission config (it sets `label-mode: script`, the class list, the
stream and the bootstrap), and split the stream across scouts with
`hawk_deploy split <config>`. For a quick first mission, build with
`--limit-images 1000`.

## Tests

```bash
uv run pytest scripts/sar_dataset
```

The tests build small fake C2A and AIDER folders and zips in the real layouts.
They cover the labeling rules, the pose mix, the full `all` pipeline from
(local) downloads to validation, that the same inputs give the same stream,
that `validate` catches changed, missing or wrongly sized tiles, that a bad
MD5 is rejected, that unsafe paths in a zip are refused, and that Hawk's own
`random` retriever maps each stream line to the right pose.
