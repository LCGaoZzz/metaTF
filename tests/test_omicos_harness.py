"""Portable harness contracts plus real installed-metaTF integration smoke tests."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

pytestmark = pytest.mark.fast
ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "agent-harness" / "omicos"
ENTRY = BUNDLE / "skills" / "metatf" / "scripts" / "run_metatf.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime():
    return load_module(ENTRY, "test_metatf_launcher")


@pytest.fixture
def installer():
    return load_module(ROOT / "agent-harness" / "install_omicos.py", "test_metatf_installer")


def fake_cpus(monkeypatch, runtime, ids):
    state = {"cpus": set(ids)}
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: state["cpus"].copy(), raising=False)

    def set_affinity(pid, cpus):
        assert set(cpus) <= state["cpus"]
        state["cpus"] = set(cpus)

    monkeypatch.setattr(os, "sched_setaffinity", set_affinity, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 224)
    for key in (*runtime._THREAD_ENV, "OMP_DYNAMIC", "MKL_DYNAMIC"):
        monkeypatch.setenv(key, "224")
    return state


def test_catalog_binding_and_resources():
    agent = (BUNDLE / "agents" / "metatf_analyst.md").read_text()
    skill = (BUNDLE / "skills" / "metatf" / "SKILL.md").read_text()
    assert "id: metatf_analyst\n" in agent
    assert "skills:\n  - metatf\n" in agent
    assert "id: metatf\nname: metatf\n" in skill
    assert "execution_mode: packaged_python\n" in skill
    assert "runtime_entrypoint: scripts/run_metatf.py\n" in skill
    assert "include_runtime_path=true" in skill
    for relative in re.findall(r"\]\((references/[^)]+)\)", skill):
        assert (BUNDLE / "skills" / "metatf" / relative).is_file()
    assert len(re.findall(r"\]\((references/[^)]+)\)", skill)) == 3
    # The executable may depend on installed metatf, never on its source checkout.
    source = ENTRY.read_text()
    assert "sys.path.insert" not in source and "parents[" not in source


@pytest.mark.parametrize("layout,prefix", [("workspace", ""), ("catalog", "domains/biology")])
def test_install_layout_and_dry_run(installer, tmp_path, layout, prefix):
    destination = tmp_path / "fresh workspace"
    planned = installer.install(destination, layout, dry_run=True)
    assert planned["status"] == "planned" and not destination.exists()
    result = installer.install(destination, layout)
    root = destination / prefix
    assert result["production_deployed"] is False
    assert result["paths"] == [str(root / "agents/metatf_analyst.md"), str(root / "skills/metatf")]
    assert (root / "skills/metatf/scripts/run_metatf.py").read_bytes() == ENTRY.read_bytes()
    assert (root / "agents/metatf_analyst.md").is_file()
    with pytest.raises(FileExistsError):
        installer.install(destination, layout)


@pytest.mark.parametrize("conflict", ["agent", "skill"])
def test_preflight_both_destinations(installer, tmp_path, conflict):
    agent = tmp_path / "agents/metatf_analyst.md"
    skill = tmp_path / "skills/metatf"
    target = agent if conflict == "agent" else skill
    target.parent.mkdir(parents=True)
    if conflict == "agent":
        target.write_text("existing agent")
    else:
        target.mkdir()
    with pytest.raises(FileExistsError):
        installer.install(tmp_path)
    assert not (skill if conflict == "agent" else agent).exists()
    if conflict == "agent":
        assert target.read_text() == "existing agent"


def test_copy_failure_rolls_back_new_content(installer, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError("simulated disk error")
    monkeypatch.setattr(installer.shutil, "copytree", fail)
    with pytest.raises(OSError, match="disk error"):
        installer.install(tmp_path)
    assert not (tmp_path / "agents/metatf_analyst.md").exists()
    assert not (tmp_path / "skills/metatf").exists()


def test_refuse_symlink_parent(installer, tmp_path):
    workspace = tmp_path / "workspace"
    elsewhere = tmp_path / "elsewhere"
    workspace.mkdir()
    elsewhere.mkdir()
    (workspace / "skills").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        installer.install(workspace)
    assert list(elsewhere.iterdir()) == []
    assert not (workspace / "agents").exists()


def test_refuse_recursive_install(installer):
    with pytest.raises(ValueError, match="own source"):
        installer.install(BUNDLE / "skills/metatf/nested", dry_run=True)


@pytest.mark.parametrize("requested,cpus,expected", [
    (None, [17, 23, 91], 3), (2, [17, 23, 91], 2),
    (224, list(range(100, 200)), 63), (None, list(range(100, 200)), 8),
])
def test_inherited_affinity_and_all_thread_limits(runtime, monkeypatch, requested, cpus, expected):
    state = fake_cpus(monkeypatch, runtime, cpus)
    result = runtime.configure_threads(requested)
    assert result["effective_threads"] == expected
    assert result["affinity_before"] == cpus
    assert state["cpus"] == set(cpus[:expected])
    assert result["affinity_after"] == cpus[:expected]
    assert result["affinity_enforced"] is True
    assert set(result["environment"].values()) == {str(expected)}


@pytest.mark.parametrize("requested", [0, -1, 1.5, True])
def test_invalid_thread_budget(runtime, requested):
    with pytest.raises(ValueError, match="positive integer"):
        runtime.configure_threads(requested)


def test_no_affinity_reports_limit_not_enforced(runtime, monkeypatch):
    fake_cpus(monkeypatch, runtime, [0, 1])
    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.delattr(os, "sched_setaffinity")
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    report = runtime.configure_threads(4)
    assert report["effective_threads"] == 2
    assert report["affinity_enforced"] is False
    assert report["affinity_after"] is None


def test_affinity_error_aborts_before_native_execution(runtime, monkeypatch):
    fake_cpus(monkeypatch, runtime, [17, 23])
    def denied(*args):
        raise PermissionError("affinity denied")
    monkeypatch.setattr(os, "sched_setaffinity", denied)
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("must not execute"))
    with pytest.raises(SystemExit) as exc:
        runtime.main(["infer-grn", "--method", "pcor", "--threads", "2"])
    assert exc.value.code == 2


@pytest.mark.parametrize("command,method", [
    ("infer-grn", "pcor"), ("infer-grn", "puic"),
    ("infer-grn", "genie3"), ("infer-grn", "sincerities"),
    ("activity", "viper"), ("activity", "aucell"),
    ("activity", "gsva"), ("activity", "ulm"),
])
def test_native_forwarding_without_scientific_reimplementation(runtime, monkeypatch, capsys, command, method):
    fake_cpus(monkeypatch, runtime, [17, 23])
    captured = {}
    def execute(executable, argv, env):
        captured.update(executable=executable, argv=argv, env=env)
    monkeypatch.setattr(os, "execve", execute)
    native = [command, "--method", method, "--input", "data with spaces.csv",
              "--output", "result.csv", "--threads=224", "--seed", "31"]
    runtime.main(native)
    assert captured["executable"] == sys.executable
    assert captured["argv"] == [sys.executable, "-m", "metatf.cli", *native, "--threads", "2"]
    assert captured["env"]["RAYON_NUM_THREADS"] == "2"
    record = json.loads(capsys.readouterr().err)
    assert record["requested_threads"] == 224 and record["effective_threads"] == 2
    assert record["argv"] == native + ["--threads", "2"]


def test_help_without_scientific_dependencies():
    result = subprocess.run([sys.executable, "-S", str(ENTRY), "--help"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "doctor|infer-grn|activity" in result.stdout


def test_copied_entrypoint_and_native_exit_status(installer, tmp_path):
    # Deliberate stand-in CLI: verifies process/packaging contracts, not numerics.
    package = tmp_path / "environment" / "metatf"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'threads': os.environ['NUMBA_NUM_THREADS']}))\n"
        "raise SystemExit(7)\n")
    workspace = tmp_path / "copied workspace"
    installer.install(workspace)
    entry = workspace / "skills/metatf/scripts/run_metatf.py"
    env = {**os.environ, "PYTHONPATH": str(package.parent)}
    result = subprocess.run([sys.executable, str(entry), "activity", "--threads", "1"],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 7
    assert json.loads(result.stdout)["threads"] == "1"
    assert json.loads(result.stderr)["event"] == "metatf_resources"


def test_doctor_explicit_missing_core(tmp_path):
    package = tmp_path / "metatf"
    package.mkdir()
    (package / "__init__.py").write_text("raise ImportError('deliberately unavailable test core')\n")
    result = subprocess.run([sys.executable, str(ENTRY), "doctor", "--threads", "1"],
                            env={**os.environ, "PYTHONPATH": str(tmp_path)},
                            text=True, capture_output=True)
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["core_importable"] is False and report["analysis_executed"] is False
    assert "deliberately unavailable" in report["error"]


@pytest.mark.parametrize("method,semantics", [("pcor", None), ("ulm", "subset"), ("ulm", "full_axis")])
def test_real_native_smoke(installer, tmp_path, method, semantics):
    metatf = pytest.importorskip("metatf")
    import numpy as np
    import pandas as pd
    expression = pd.DataFrame(np.random.default_rng(42).integers(1, 15, size=(9, 12)),
                              index=[f"g{i}" for i in range(9)], columns=[f"c{i}" for i in range(12)])
    network = pd.DataFrame({"tf": ["A"] * 5 + ["B"] * 4,
                            "target": [f"g{i}" for i in range(9)],
                            "weight": [1, -2, 3, 0.5, -1, 1, 2, -1, 3]})
    expression.to_csv(tmp_path / "exp.csv")
    network.to_csv(tmp_path / "net.csv", index=False)
    installer.install(tmp_path / "workspace")
    entry = tmp_path / "workspace/skills/metatf/scripts/run_metatf.py"
    output = tmp_path / "result.csv"
    command = [sys.executable, str(entry), "infer-grn" if method == "pcor" else "activity",
               "--method", method, "--input", str(tmp_path / "exp.csv"),
               "--output", str(output), "--threads", "1"]
    if method == "ulm":
        command += ["--network", str(tmp_path / "net.csv"), "--minsize", "3", "--semantics", semantics]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stderr.splitlines()[0])
    assert report["effective_threads"] == 1
    if method == "pcor":
        expected = metatf.infer_grn(expression, method="pcor", threads=1)
    else:
        expected, pvalue = metatf.regulon_activity(expression, network=network, method="ulm",
                                                  minsize=3, semantics=semantics, threads=1)
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "result_pval.csv", index_col=0),
                                      pvalue, check_names=False, rtol=1e-9, atol=1e-10)
    pd.testing.assert_frame_equal(pd.read_csv(output, index_col=0), expected,
                                  check_names=False, rtol=1e-9, atol=1e-10)


def test_real_doctor():
    metatf = pytest.importorskip("metatf")
    result = subprocess.run([sys.executable, str(ENTRY), "doctor", "--threads", "1"],
                            text=True, capture_output=True)
    report = json.loads(result.stdout)
    assert result.returncode == 0, report
    assert report["core_importable"] is True
    assert report["analysis_executed"] is False
    assert report["metatf_version"] == metatf.__version__
    assert set(report["rust_importable"]) == {"rust_pcor_v2", "rust_puic", "rust_genie3"}
