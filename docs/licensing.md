# Licensing and distribution

Original project code is licensed under [Apache-2.0](../LICENSE).
The project attribution is recorded in [NOTICE](../NOTICE).
Third-party files and dependencies retain their own licenses; the project license
does not replace them. In particular, the bundled robot model and meshes have a
separate [license](../packages/soarm/soarm/robot/LICENSE) and
[provenance record](../packages/soarm/soarm/robot/README.md).

Apache-2.0 requires preserving applicable copyright and attribution notices when
redistributing the work, providing the license, and identifying modified files.
A downstream distribution must retain applicable NOTICE attribution in a form
allowed by the license. This does not require a logo, a permanent UI credit, or a
research citation. See [Apache-2.0, section 4](https://www.apache.org/licenses/LICENSE-2.0).

## Dependencies

The declarations in [soarm](../packages/soarm/pyproject.toml),
[soarm-lab](../packages/soarm_lab/pyproject.toml), and the
[workspace](../pyproject.toml) define which packages are installed.
[uv.lock](../uv.lock) records resolved versions across extras and platforms;
presence in that file does not mean a package is part of every installation.

| Installation | Dependencies and upstream licenses |
| --- | --- |
| `soarm` | `machinekind-twin-api`: Apache-2.0; its dependency NumPy: BSD-3-Clause. |
| `soarm[feetech]` | pyserial: BSD-3-Clause. This backend uses the serial protocol directly and does not install the Feetech SDK. |
| `soarm[panel]` | pyserial: BSD-3-Clause; PyYAML: MIT. |
| `soarm[lerobot]` | LeRobot: Apache-2.0; its `feetech` extra adds Feetech Servo SDK: Unlicense, deepdiff: MIT, and pyserial. Requires Python 3.12 or newer. |
| `soarm-lab` | `soarm` and `machinekind-twin-api`: Apache-2.0; NumPy: BSD-3-Clause; PyYAML: MIT; MuJoCo: Apache-2.0; Gymnasium: MIT. |
| `soarm-lab[train]` | MuJoCo Warp and JAX: Apache-2.0; includes Warp and JAX's own dependencies. |
| `soarm-lab[torch]` | PyTorch: BSD-3-Clause; platform-specific binary dependencies have additional terms. |
| Workspace `twin` group | `machinekind-twin`: Apache-2.0; adds NumPy, PyYAML, OpenCV contrib, and the API package. |

The API and twin application both declare Apache-2.0 at the
[pinned twin revision](https://github.com/machinekind/twin/tree/941fc6b170aaa24b110c1ca31e1b7ddbca2515c0).
The API package includes the license text. The workspace fetches it from this
public Git repository; it is not currently published on PyPI. Follow the
[workspace installation instructions](../README.md#get-started) to use that source.

Primary license references: [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt),
[pyserial](https://github.com/pyserial/pyserial/blob/master/LICENSE.txt),
[PyYAML](https://github.com/yaml/pyyaml/blob/main/LICENSE),
[LeRobot](https://github.com/huggingface/lerobot/blob/main/LICENSE),
[Feetech Servo SDK 1.0.0](https://pypi.org/project/feetech-servo-sdk/1.0.0/),
[deepdiff](https://github.com/seperman/deepdiff/blob/master/LICENSE),
[MuJoCo](https://github.com/google-deepmind/mujoco/blob/main/LICENSE),
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE),
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp/blob/main/LICENSE),
[JAX](https://github.com/jax-ml/jax/blob/main/LICENSE), and
[PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE).
Use the license files supplied with the actual released artifacts when distributing them.

## What a release includes

The project wheels declare separately installed Python dependencies; they do not
bundle those libraries merely by listing them in package metadata. Distributing
these wheels or the project source therefore differs from distributing a complete
environment, container image, installer, or executable that contains dependencies.
Keep the licenses and notices for the project and its bundled robot assets in
project releases. Collect additional licenses, notices, and required source or
source-access information for dependencies actually included in a larger distribution.

Both Python package roots contain copies of the repository's `LICENSE` and
`NOTICE`; keep those copies in sync. Each package's `license-files` configuration
includes its notices in wheels and source distributions.

Some optional dependency paths require particular attention:

- LeRobot brings in [certifi](https://github.com/certifi/python-certifi/blob/master/LICENSE)
  (MPL-2.0) and [tqdm](https://github.com/tqdm/tqdm/blob/master/LICENCE)
  (MPL-2.0 and MIT). Preserve their notices and meet MPL requirements for the
  covered files, including any distributed modifications. Independent project
  files do not become MPL-licensed merely because these packages are installed.
- OpenCV wheels used by LeRobot or the optional twin can contain FFmpeg
  (LGPL-2.1); non-headless Linux wheels can also contain Qt (LGPL-3).
  Check the exact wheel's third-party license file and comply with the applicable
  source and relinking requirements when shipping these binaries.
  See the [OpenCV packaging guidance](https://github.com/opencv/opencv-python#licensing).
- Prebuilt Warp packages include libmathdx under separate NVIDIA terms, and
  Linux PyTorch installations can pull in CUDA and other NVIDIA libraries.
  Preserve applicable notices and verify redistribution permissions for the
  exact components shipped. See [Warp's licensing notes](https://github.com/NVIDIA/warp#license)
  and the [CUDA license agreement](https://docs.nvidia.com/cuda/eula/index.html).

This is distribution guidance based on the declared dependencies and checked
licenses, not certification of every transitive package or binary build. Recheck
the installed artifacts when changing versions, extras, platforms, or packaging.
