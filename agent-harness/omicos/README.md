# Omicos integration

One Agent (`metatf_analyst`) bound to one Skill (`metatf`), using the same
portable Agent/Skill layout as the iobrx integration. The runtime is a thin
launcher for the installed `metatf` package, not a second implementation.
There are no host-specific paths, tokens or additional runtime dependencies.

## LLM-led input preparation

The Agent/Skill instructs the LLM to inspect expression scale and available
slots, select the method-appropriate data, perform necessary preparation with
Omicos's existing Python/shell tools, and then run metaTF. Routine preparation
does not require step-by-step user approval. Suitable processed data is reused;
material unresolved ambiguity is surfaced instead of guessed away. See the
[Skill](skills/metatf/SKILL.md) and its input/method references.

This is guidance for the Agent, not a built-in classifier or normalizer in
`run_metatf.py`. Direct CLI calls still use exactly the input supplied (native
h5ad reads `X`); they do not gain the LLM's assessment/preparation behavior.

## Install into a workspace

From a metaTF-py checkout:

```bash
python agent-harness/install_omicos.py --destination /path/to/analysis-workspace --dry-run
python agent-harness/install_omicos.py --destination /path/to/analysis-workspace
```

This copies `agents/metatf_analyst.md` and the entire `skills/metatf/` directory.
Both destinations are preflighted; existing files, directories or symlinks
are not replaced. Copy failures remove newly created Agent/Skill content.
The installer does not install Python packages or change Omicos settings.
The copied Skill must remain intact, but the original checkout can be removed.

Workspace extension discovery is subject to the host's account/workspace
capabilities. Copying files is not production registration or an entitlement
bypass. Following the iobrx local-development convention, authorized maintainers
can configure a local test core with host-native paths:

```bash
export OMICOS_TEMPLATES_DIR=/path/to/analysis-workspace
export OMICOS_SKILL_ROOTS=/path/to/analysis-workspace/skills
export OMICOS_AGENTS_OFFLINE=1
export OMICOS_SKILLS_OFFLINE=1
```

Verify `/api/agents` exposes `metatf_analyst` and `/api/skills` exposes `metatf`.
Resolve `scripts/run_metatf.py` through `skill_resource` with
`include_runtime_path: true`, then use the prepared analysis interpreter.
These are live host checks, not results implied by this repository's tests.

## Shared biology catalog

On a clean feature branch of an omicos-admin checkout:

```bash
python agent-harness/install_omicos.py --destination /path/to/omicos-admin --layout catalog
cd /path/to/omicos-admin
python scripts/validate_skills.py --skill metatf
```

The installed paths are:

```text
domains/biology/agents/metatf_analyst.md
domains/biology/skills/metatf/SKILL.md
domains/biology/skills/metatf/scripts/run_metatf.py
domains/biology/skills/metatf/references/*.md
```

Review a separate admin PR and follow the host's normal catalog deployment
procedure. This metaTF-py bundle does not alter that repository or deploy an
Agent to a live service. Do not use legacy top-level catalog directories.

## Execution and verification

```bash
python agent-harness/omicos/skills/metatf/scripts/run_metatf.py doctor --threads 4
python agent-harness/omicos/skills/metatf/scripts/run_metatf.py infer-grn --help
python agent-harness/omicos/skills/metatf/scripts/run_metatf.py activity --help
python -m pytest tests/test_omicos_harness.py -q
```

The tests cover resource limits, native argument forwarding, error propagation,
portable copying, overwrite refusal, resource references and real installed
metaTF PCOR/ULM smoke calls. They use the existing `fast` marker, so the existing
Python/Rust CI jobs discover them without another workflow or dependency.
Numerical smoke checks test the integration, not parity against R. Live Omicos
catalog loading and backend benchmarks require their actual environments.

Reviewed source anchors: metaTF-py `0ec381c9020cf62723b425ca0cf5f669d42801cf`
(`src/metatf/{api,cli,adapters,_rust}.py`); iobrx integration
[`5ca23a985d722ce346e79846a90e6e1b9161e179`](https://github.com/LCGaoZzz/iobrx/tree/5ca23a985d722ce346e79846a90e6e1b9161e179/agent-harness/omicos).
