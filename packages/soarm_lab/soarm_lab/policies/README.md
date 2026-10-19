# Bundled policy provenance

`reach-v3` and `lift-v3` come from the SO-101 PPO training work in the
[Machinekind digital twin](https://github.com/machinekind/twin). The training
code uses MuJoCo simulation and domain randomization. Git records both checkpoint
additions under the author `przemeknowak781`, dated 2026-09-24.

## Upstream checkpoints

| Policy | Original checkpoint | Training history recorded by the commit |
| --- | --- | --- |
| `lift-v3` | [`assets/policies/lift-v3/policy.pt`](https://github.com/machinekind/twin/blob/3c3bc2808151d2f6c430d2d61c22878e30266116/assets/policies/lift-v3/policy.pt) | [Fine-tuning update](https://github.com/machinekind/twin/commit/3c3bc2808151d2f6c430d2d61c22878e30266116) that added `lift-v3` and support for saving and restoring critic weights. |
| `reach-v3` | [`assets/policies/reach-v3/policy.pt`](https://github.com/machinekind/twin/blob/13cfb693119b3929dab35b0fa9d4d013cf857a39/assets/policies/reach-v3/policy.pt) | [Fine-tuned from `reach-v2`](https://github.com/machinekind/twin/commit/13cfb693119b3929dab35b0fa9d4d013cf857a39) with a penalty for wrist rotation away from the home pose. |

The [reach metadata](reach-v3/meta.json) and [lift metadata](lift-v3/meta.json)
record task settings, network dimensions, training steps, randomization and
evaluation results. Their `notes` fields identify the preceding checkpoint by
its creation timestamp. These files match the metadata at the respective
upstream commits. They are training summaries, not a complete record of every
earlier training run.

## Import and format conversion

[Commit `5acba3408df83eb26a4c10e5276b0a83c10b76fa`](https://github.com/machinekind/so-arm101/commit/5acba3408df83eb26a4c10e5276b0a83c10b76fa)
imported both policy folders from the twin into this package. The imported
`policy.pt` and `meta.json` files match the upstream checkpoints above byte for
byte.

[Commit `c1c5b29ada65168c2f5f194dddf842480679e88b`](https://github.com/machinekind/so-arm101/commit/c1c5b29ada65168c2f5f194dddf842480679e88b)
converted the PyTorch checkpoints to `policy.npz` for NumPy inference, retaining
the actor, critic and observation-normalization arrays. The bundled NPZ files
are unchanged from that conversion, and the metadata is unchanged from the
original import. The [policy API](../rl/policy.py) documents the NPZ format.

The policies are distributed under the license stated in the
[soarm-lab README](../../README.md#license).
