# so-arm101

Hardware control, simulation and reinforcement learning for the SO-101 (SO-ARM101)
robot arm with Feetech STS3215 servos. Use it to move a physical arm from a browser,
evaluate manipulation policies in MuJoCo, and train policies for reaching, lifting
and pick-and-place tasks.

The repository is a [uv](https://docs.astral.sh/uv/) workspace with two Python packages:

- **[soarm](packages/soarm/README.md)** provides the robot model, Feetech and LeRobot
  hardware backends, a browser control panel and a serial-to-TCP bridge.
- **[soarm-lab](packages/soarm_lab/README.md)** provides simulation environments,
  PPO training, policy evaluation, bundled reach and lift policies, and inverse kinematics.

Both integrate with the [Machinekind digital twin](https://github.com/machinekind/twin)
as plugins. The twin measures the physical setup and runs policies and inverse
kinematics through its safety supervisor. The lab can also run independently using
its bundled simulated setup.

## Get started

Requires Git, Python 3.10+ and uv. The workspace fetches the shared plugin API
from a pinned commit of the public [twin repository](https://github.com/machinekind/twin).
The API package is not currently published on PyPI.

```bash
git clone https://github.com/machinekind/so-arm101.git
cd so-arm101
uv sync
```

### Control an arm

Read the [safety rules](docs/safety.md) and [first-contact guide](docs/first-contact.md)
before powering the arm. Connect USB, then motor power, and launch the panel:

```bash
uv run soarm-panel
```

The panel detects a single USB-serial device and opens joint and gripper controls
in your browser, including Home, STOP and torque controls. If several devices are
connected, specify one with `--port /dev/ttyACM0` or `--port COM5`.

### Run a simulation

List available policies and evaluate the bundled lift policy without hardware:

```bash
uv run soarm-lab policies
uv run soarm-lab eval lift-v3
```

Evaluation uses the bundled MuJoCo station by default. To use a measured setup,
export it from the twin with `twin export ~/stations/desk1`, then pass
`--station ~/stations/desk1` to the lab's training or evaluation command.

GPU training is PPO in JAX on MuJoCo Warp and requires CUDA. See the
[lab documentation](packages/soarm_lab/README.md) for training dependencies,
fine-tuning and policy formats. Policy inference uses NumPy and requires neither JAX
nor PyTorch. PyTorch is needed only to convert checkpoints of the old PyTorch trainer.
One lift policy can also [learn several wooden blocks](packages/soarm_lab/README.md#lift-a-block).

### Run Dream-1 on the arm

[Dream Labs](https://dreamlabs.ai)' Dream-1 model drives the arm from a text prompt and
the cameras. It needs a Dream Labs API key and the `dream` extra. See the
[lab documentation](packages/soarm_lab/README.md#dream-labs) for the setup and the caveats.

```bash
uv sync --extra dream
uv run soarm-lab dream models
uv run soarm-lab dream arm --prompt "pick up the cube" --camera scene=0 --camera wrist=1
```

`uv run soarm-zero` moves an arm slowly to its zero pose and holds it; `--off` releases torque.

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run pyright
```

Optional dependencies: `uv sync --extra train` installs JAX and MuJoCo Warp for training;
`uv sync --group torch` installs PyTorch for converting old checkpoints;
`uv sync --group twin` installs the twin for integration tests;
`uv sync --extra dream` installs the Dream Labs client for `soarm-lab dream`.
Tests requiring unavailable optional dependencies, GPU support or rendering skip automatically.

See [conventions](docs/conventions.md) for units, joint names, package boundaries
and test conventions.

## License

[Apache-2.0](LICENSE), with project attribution in [NOTICE](NOTICE). The SO-101
MuJoCo model and meshes are also Apache-2.0; see their
[third-party notices](packages/soarm/THIRD_PARTY_NOTICES).

See [licensing and redistribution](docs/licensing.md) for dependency licenses
and how to preserve notices when distributing packages or bundled applications.
