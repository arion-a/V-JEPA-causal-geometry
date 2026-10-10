# Gate 0 report - V-JEPA 2.1 ViT-L on the pod

Status: **PASS** (all Gate 0 items closed; one caveat noted). Evidence: `results/jobs/001_gate0/`, `results/jobs/002_gate0b/` (both pushed by the pod worker), ledger entries 12-14.

## Verified facts
| Item | Value |
|---|---|
| Official repo | facebookresearch/vjepa2 @ `204698b45b3712590f06245fbfba32d3be539812`, MIT license |
| Entry point | `vjepa2_1_vit_large_384` (also base/giant/gigantic 2.1 variants exist) |
| Checkpoint | `vjepa2_1_vitl_dist_vitG_384.pt`, 5,151,198,524 bytes, **SHA256 `7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6`** (downloaded from the official host after patching the repo's placeholder `VJEPA_BASE_URL=http://localhost:8300`) |
| Model | ViT-L, 304.7M params, 24 blocks, width 1024, 16 heads; patch 16, tubelet 2; native 64 frames @ 384x384; also has an image patch embed (tubelet 1) |
| Tokens | 16-frame clip @384 -> 8 x 24 x 24 = 4608 tokens/block; order (t,h,w) confirmed (see below) |
| Hooks | 24 block outputs `[1,4608,1024]`; fp32 repeat runs identical (max abs diff 0.0) |
| Hardware | RTX 4000 Ada 20 GB, torch 2.8.0+cu128; peak memory 2.5-3.7 GB |

## Layers the model itself exposes (read from `app/vjepa_2_1/models/vision_transformer.py`)
ViT-L `hierarchical_layers = [5, 11, 17, 23]` (0-indexed blocks), each with its own LayerNorm in `norms_block`. Default forward returns `norms_block[-1](block23)`; with `return_hierarchical=True` it returns the four normed intermediate outputs (the deep self-supervision targets). Our raw hooks tap the pre-norm residual stream of all 24 blocks.

## Findings that change the plan
1. **Use fp32 for extraction.** bf16 differs from fp32 by about 1.0-1.75% relative at every block (not under the 0.1% proposed). Perturbation signals are probably smaller. fp32 cost is small: 64-frame clip about 5 s (0.2 clips/s), 16-frame about 0.55 s/clip (1.8 clips/s at batch 4); 200 clips of 64 frames about 17 minutes.
2. **Object localisation exists only in early layers.** A static bright square covering 6.2% of tokens holds 57% of the change mass at block 0 (about 20x inside/outside) and 67% at block 1, but 14% at block 7, 8% at block 11 and about 6.4% (uniform) from block 17. So object-window pooling is meaningful for early/middle layers; deep layers spread object information across all tokens, so global pooling must be a primary analysis, not only a sensitivity check. (One test, one large static object; not a general claim.)
3. **Analysis layers.** Proposed primary set: the four hierarchical layers {5,11,17,23} with their own norms (the model's trained tap points); all 24 raw block outputs as exploratory.
4. **Input format.** Native is 64 frames @384; 16 frames also runs. Simulator must render at 384 (or resize, logged). Decide frames per clip before Stage 1.
5. The weights-host HEAD probe returned 403 because it used a nonexistent filename; the real download succeeded, so it is not evidence of a problem.

## Caveat
The token-map test used one object geometry and the difference-norm metric; ordering is confirmed, finer claims are not.

## Next
Stage 1 (simulator + 200-clip manifest, rendering at 384) and Gate 1 (20-clip smoke with noise floor in fp32). Decisions needed: frames per clip (16 vs 64), primary representation (global mean vs early-layer object window), primary layer set.
