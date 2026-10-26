# Search-and-rescue mission data for Hawk (C2A + AIDER)

`build_sar_mission.py` merges two public datasets into one Hawk mission:
tiles with exactly one label each, a shuffled stream file, a bootstrap zip and
a config snippet.

## 1. Download the datasets

AIDER (Kyrkou and Theocharides, CC BY 4.0, about 276 MB):
https://zenodo.org/records/3888300 → `AIDER.zip`

C2A (Nihal et al., 2024), from the authors' repository
https://github.com/Ragib-Amin-Nihal/C2A, which links Google Drive and Kaggle
(https://www.kaggle.com/datasets/rgbnihal/c2a-dataset). The repository does not
state a license, so check with the authors or Jan before sharing it further.

Unzip both. The script expects:

```
<c2a-root>/                         # the "new_dataset3" folder
    train/images/  val/images/  test/images/
    All labels with Pose information/   # YOLO lines: class xc yc w h pose
<aider-root>/
    .../normal/ ...                 # one folder per AIDER class
```

## 2. Check the numbers first (writes no images)

```bash
python scripts/sar_dataset/build_sar_mission.py \
    --c2a-root ~/data/C2A/new_dataset3 --aider-root ~/data/AIDER \
    --out ~/data/hawk_sar --dry-run
```

Read `~/data/hawk_sar/stats.json`. The numbers that matter:

- `c2a_tile_outcomes`: how many tiles had one pose, were empty, had mixed
  poses, or only cut-off people. If most tiles are `mixed_pose`, rerun with
  `--mixed majority` or a smaller `--tile-size`.
- `c2a_tiles_by_class`: how rare each pose really is. This replaces the
  "upright common, lying rare" assumption on the slides.
- `stream_positive_rate`: the share of tiles with a person. Set it with
  `--positive-rate` (default 0.05). Try two or three values (the merge-ratio
  risk on the slides).

### What the real data looks like (dry runs, November 3)

| Setting | Single-pose tiles | Mixed-pose tiles (dropped) | Empty tiles | Negative tiles in total |
|---|---|---|---|---|
| 256 px tiles | 7,886 | 52,300 | 13,422 | 35,343 |
| 128 px tiles | 54,281 | 93,237 | 92,978 | 163,566 |

C2A packs about 35 people of mixed poses into each image, so 256 px tiles are
mostly mixed. 128 px tiles give about 54,000 clean single-pose tiles, so use
`--tile-size 128`.

C2A's five poses are roughly balanced (about 10,000 to 13,000 single-pose tiles
each at 128 px). No pose is naturally rare. To make one rare, set the mix
yourself with `--pose-mix`, for example
`--pose-mix upright=0.4,sitting=0.25,bent=0.15,kneeling=0.15,lying=0.05`.

## 3. Build it

Same command without `--dry-run`. Writing tiles can take a while; with `--max-seconds N` the script stops after N seconds and the same command continues where it left off. For a quick first mission, add
`--limit-images 1000` (the 1,000-image fallback from the roadmap).

Then copy `~/data/hawk_sar` to the scout's data root, paste
`mission_snippet.yml` into the mission config, and split the stream across
scouts with `hawk_deploy split <config>`.

## How tiles are labeled

Hawk's `dnn_classifier` gives one class per tile. Boxes with real coordinates
are all stored as class 1 by the scout's data manager, so the five poses would
collapse into one class if C2A were fed in with its boxes. The script cuts every
image into a grid of `--tile-size` tiles and labels each tile:

| Tile contents | Label |
|---|---|
| No person | negative (0) |
| People of one pose, at least `--min-visible` of each box inside | that pose (1 to 5) |
| People of different poses | dropped, or the majority pose with `--mixed majority` |
| Only cut-off people | dropped (we can't tell what it shows) |

Labels: 0 negative, 1 bent, 2 kneeling, 3 lying, 4 sitting, 5 upright, the order
of `dataset.class_list` in the snippet.

AIDER tiles come only from the `normal` folder by default (`--aider-classes`).
C2A pasted its people onto AIDER's disaster photos, so adding those disaster
photos back without people could let the model learn to spot the pasting
instead of the person. Empty C2A tiles are also used as negatives
(`--no-c2a-empty-negatives` turns that off).

The bootstrap takes `--bootstrap-per-pose` tiles per pose from C2A's train
split and `--bootstrap-negatives` AIDER tiles. Every tile from a bootstrap
source image is kept out of the stream.

## Tests

```bash
uv run pytest scripts/sar_dataset
```
