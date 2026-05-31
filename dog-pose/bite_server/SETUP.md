# BITE — runtime decision + spike setup (Phase 3, Chunk A)

Chunk A spike from [PLAN-phase3.md](../../PLAN-phase3.md): get a **first 3D dog
mesh** on our test photos and judge fit quality before building anything.
Throwaway validation — nothing here is production.

## Runtime decision (locked for the spike): Google Colab (GPU)

Per Plan Erratum 1, BITE's *documented* stack (`torch 1.6 / pytorch3d 0.2.5 /
cu101`, 2020) won't resolve on this arm64 (M4) Mac. Colab is a Linux+GPU box and
the fastest path to a first mesh. License is **non-commercial research only** —
fine for this project (research/personal).

## ⭐ The shortcut: don't reproduce the 2020 stack — use camenduru's recipe

`camenduru/bite-colab` already solved the env the easy way: a **patched `dev`
fork** that runs on **modern torch 1.13 + pytorch3d @stable**, and it pulls the
otherwise registration-gated model files straight from the HF Space over
git-LFS — **no MPI download/registration needed**. This is the primary path.

- Ready-made notebook: <https://github.com/camenduru/bite-colab> (`bite_colab.ipynb`)
- Our self-contained copy (same recipe + saves outputs): `colab/BITE_spike.ipynb`

The working recipe (verbatim from camenduru — deps live in the fork's own
patched `requirements.txt`; assets are bundled in a public HF repo, no LFS dance
and no registration):
```bash
git clone -b dev https://github.com/camenduru/bite_gradio-hf
git clone https://huggingface.co/camenduru/bite          # bundles checkpoint/ data/ datasets/
mv bite/checkpoint bite_gradio-hf/checkpoint
mv bite/data       bite_gradio-hf/data
mv bite/datasets   bite_gradio-hf/datasets
cd bite_gradio-hf && pip install -r requirements.txt
python ./scripts/gradio_demo.py        # interactive; or the batch script below
```
Working dir is `bite_gradio-hf`.

## Validating quality (two ways, easiest first)

1. **Gradio app** — `scripts/gradio_demo.py` gives an upload-an-image UI. Drop
   each test photo in, eyeball the recovered 3D mesh. Fastest qualitative check.
   (The hosted HF Space `runa91/bite_gradio` itself is **offline** — no GPU — so
   run the app yourself in Colab.)
2. **Batch over crops** — BITE has no detector and wants the dog centred. The
   notebook crops our photos with YOLOv8n-seg (same model the DLC server uses)
   into `bite/datasets/test_image_crops/`, then runs:
   ```bash
   python scripts/full_inference_including_ttopt.py --workers 4 \
     --config refinement_cfg_test_withvertexwisegc_csaddnonflat_crops.yaml \
     --model-file-complete cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar \
     --suffix ttopt_spike
   ```
   ttopt is iterative — **time it** (seconds-to-minutes/image); the number sizes
   the Chunk-B precompute/caching design.

### Test images (the spike set)
- `Bounding box efforts resolution too large.png` (repo root) — hard, small-in-frame.
- `dog_pose_composite first working.png` (repo root) — clean subject.
- `/Users/tjlefebvre/Desktop/image_proxy.webp` — Phase 2 fixture.
- One clear side-on photo close to the reference sketches.

## Fallback only — faithful 2020 conda env

If camenduru's `dev` fork ever breaks, reproduce BITE's exact env per the
official README: `conda create -n conda_bite python=3.7.6 ipython` →
`conda install pytorch==1.6.0 torchvision cudatoolkit=10.1 -c pytorch` →
`pip install -r requirements.txt` (this is where `pytorch3d==0.2.5` must build —
the hard part). Gated assets via the official link
<https://owncloud.tuebingen.mpg.de/index.php/s/BpPWyzsmfycXdyj> (`checkpoint/`
→ repo root, `data/` subfolders → `data/`). StanfordExtra is **not** needed for
our own crops. May need to edit `src/configs/dataset_path_configs.py`.

## Success criteria for Chunk A

Passes iff: (1) a runtime actually produced a fit; (2) the mesh roughly conforms
to each dog (limbs roughly placed) on at least the two clean photos; (3)
per-image wall-clock recorded. Put overlay screenshots + a one-paragraph verdict
in `spike_outputs/` and report back. **Do not** start Chunk B (`bite_server`)
until this passes.
