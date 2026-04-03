#!/usr/bin/env python3
"""
Benchmark: incremental build comparison for go2nix modes.

Measures rebuild time after touching a single file at different dep-graph
depths and with different edit types (private vs exported symbols).

Usage:
    python tests/bench-incremental.py [--runs N] [--scenario S] [--tools nix,nix-ca]

Requires: nix with go2nix-nix-plugin, socat (for CA mode).
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class BenchmarkResult:
    scenario: str
    tool: str
    times: list[float] = field(default_factory=list)
    builds: list[int] = field(default_factory=list)

    @property
    def builds_mean(self) -> float:
        return statistics.mean(self.builds) if self.builds else 0.0

    @property
    def builds_max(self) -> int:
        return max(self.builds) if self.builds else 0

    @property
    def mean(self) -> float:
        return statistics.mean(self.times) if self.times else 0.0

    @property
    def stddev(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def min(self) -> float:
        return min(self.times) if self.times else 0.0

    @property
    def max(self) -> float:
        return max(self.times) if self.times else 0.0


# Touch points at different dep-graph depths.
# Paths are relative to the fixture root (torture-project).
TOUCH_SCENARIOS = {
    # Leaf: only main depends on nothing else. Cascade = main + link.
    "leaf": "app-full/cmd/app-full/main.go",
    # Mid: aws is imported by main only. Cascade = aws + main + link.
    "mid": "internal/aws/aws.go",
    # Deep: common is imported by ~9 local modules + main.
    # Cascade = common + all dependents + main + link.
    "deep": "internal/common/common.go",
}

GO_DIR = "app-full"
MOD_ROOT = "app-full"
NIX_EXPR_TEMPLATE = """\
{{ srcPath ? {fixture_path} }}:
let
  pkgs = import <nixpkgs> {{ system = "{system}"; }};
  go2nixLib = import {go2nix_src}/lib.nix {{}};
  goEnv = go2nixLib.mkGoEnv {{
    go = pkgs.go_1_26;
    go2nix = import {go2nix_src}/packages/go2nix {{ inherit pkgs; }};
    inherit (pkgs) callPackage;
  }};
in
goEnv.buildGoApplication {{
  src = srcPath;
  modRoot = "{mod_root}";
  goLock = "${{srcPath}}/{mod_root}/go2nix.toml";
  pname = "torture-bench";
  version = "0.0.1";
  subPackages = [ "./cmd/app-full" ];
  doCheck = false;
  {extra_attrs}
}}
"""

_TOUCH_MARKER = "// BENCHMARK_TOUCH"

# Fixed symbol names + rotating values: each touch updates an existing
# symbol's body rather than declaring a fresh one. Models the common
# dev-loop edit ("changed a constant"). Declaring a new symbol each
# time would also work for `private` (Go's export data only encodes
# inittask presence, not body), but the symbol-name-in-export-data
# effect would muddy `exported`.
_TOUCH_TEMPLATES = {
    "private": "var _benchTouch = uint64({ts}) {marker}\n",
    "exported": "var BenchTouch = uint64({ts}) {marker}\n",
}


def get_repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[float, str, str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=cwd, env=full_env, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"  COMMAND FAILED (exit {result.returncode}):")
        print(f"  stderr: {result.stderr[-500:]}")

    return elapsed, result.stdout or "", result.stderr or ""


def touch_file(path: Path, mode: str) -> None:
    if not path.exists():
        print(f"  Warning: file not found: {path}")
        return
    content = path.read_text()
    ts = time.time_ns()
    line = _TOUCH_TEMPLATES[mode].format(ts=ts, marker=_TOUCH_MARKER)
    path.write_text(f"{content}\n{line}")


def restore_file(path: Path) -> None:
    if not path.exists():
        return
    lines = path.read_text().splitlines(keepends=True)
    kept = [ln for ln in lines if _TOUCH_MARKER not in ln]
    if len(kept) != len(lines):
        if kept and kept[-1] == "\n":
            kept.pop()
        path.write_text("".join(kept))


class LocalDaemon:
    """Manages a local nix daemon via socat for CA derivation support."""

    def __init__(self, tmpdir: Path, extra_features: str = ""):
        self.store_root = tmpdir / "ca-store"
        self.socket = self.store_root / "daemon.sock"
        self.pid: int | None = None
        self.features = f"nix-command {extra_features}".strip()

    def start(self) -> None:
        # Fresh store each run to avoid stale state.
        if self.store_root.exists():
            shutil.rmtree(self.store_root, ignore_errors=True)
        self.store_root.mkdir(parents=True, exist_ok=True)

        daemon_sh = self.store_root / "daemon.sh"
        daemon_sh.write_text(
            f"#!/usr/bin/env bash\n"
            f'exec nix daemon --stdio \\\n'
            f'  --option experimental-features "{self.features}" \\\n'
            f'  --option sandbox false \\\n'
            f'  --option allow-import-from-derivation true \\\n'
            f'  --store "local?root={self.store_root}"\n'
        )
        daemon_sh.chmod(0o755)

        proc = subprocess.Popen(
            ["socat", f"UNIX-LISTEN:{self.socket},fork", f"EXEC:{daemon_sh}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.pid = proc.pid
        time.sleep(1)

        # Verify the socket exists.
        if not self.socket.exists():
            raise RuntimeError(f"Local daemon failed to start (no socket at {self.socket})")
        print(f"  Local daemon: PID {self.pid}, store {self.store_root}")

    def stop(self) -> None:
        if self.pid:
            try:
                os.kill(self.pid, 15)
            except ProcessLookupError:
                pass
            self.pid = None

    @property
    def env_remote(self) -> str:
        return f"unix://{self.socket}?root={self.store_root}"


class NixTool:
    def __init__(
        self,
        name: str,
        nixpkgs_path: str,
        plugin_path: str,
        gomodcache: str,
        expr_path: str,
        extra_opts: list[str] | None = None,
        daemon: LocalDaemon | None = None,
    ):
        self.name = name
        self.nixpkgs_path = nixpkgs_path
        self.plugin_path = plugin_path
        self.gomodcache = gomodcache
        self.expr_path = expr_path
        self.extra_opts = extra_opts or []
        self.daemon = daemon

    def build(self, src_path: str | None = None) -> tuple[float, int]:
        cmd = [
            "nix-build",
            "-I",
            f"nixpkgs={self.nixpkgs_path}",
            "--option",
            "plugin-files",
            self.plugin_path,
            "--option",
            "allow-import-from-derivation",
            "true",
            *self.extra_opts,
            self.expr_path,
            "--no-out-link",
        ]
        if src_path:
            cmd.extend(["--arg", "srcPath", src_path])

        env: dict[str, str] = {"GOMODCACHE": self.gomodcache}
        if self.daemon:
            env["NIX_REMOTE"] = self.daemon.env_remote
        elapsed, stdout, stderr = run_command(cmd, env=env)
        # Count the per-derivation `building '/nix/store/...'` lines.
        # In CA mode each drv prints one "building" then one "resolved
        # derivation" — we count "building" only.
        built = (stdout + stderr).count("building '/nix/store/")
        return elapsed, built


def resolve_paths(repo_root: Path) -> tuple[str, str, str]:
    """Resolve nixpkgs path, plugin path, and gomodcache."""
    _, nixpkgs_path, _ = run_command(
        ["nix", "eval", "--raw", "nixpkgs#path"]
    )
    _, plugin_out, _ = run_command(
        [
            "nix",
            "build",
            f"{repo_root}#go2nix-nix-plugin",
            "--no-link",
            "--print-out-paths",
        ]
    )
    plugin_path = f"{plugin_out.strip()}/lib/nix/plugins/libgo2nix_plugin.so"

    # Use the pre-built gomodcache from the benchmark package if available,
    # otherwise fall back to user's GOMODCACHE.
    gomodcache = os.environ.get("GOMODCACHE", "")
    if not gomodcache:
        _, gomodcache, _ = run_command(["go", "env", "GOMODCACHE"])
        gomodcache = gomodcache.strip()

    return nixpkgs_path.strip(), plugin_path, gomodcache


def write_nix_expr(
    tmpdir: Path,
    name: str,
    fixture_path: str,
    go2nix_src: str,
    system: str,
    extra_attrs: str = "",
) -> str:
    content = NIX_EXPR_TEMPLATE.format(
        fixture_path=fixture_path,
        go2nix_src=go2nix_src,
        system=system,
        mod_root=MOD_ROOT,
        extra_attrs=extra_attrs,
    )
    path = tmpdir / f"bench-{name}.nix"
    path.write_text(content)
    return str(path)


def run_touch_benchmark(
    tools: list[NixTool],
    fixture_copy: Path,
    scenario: str,
    touch_mode: str,
    runs: int,
) -> list[BenchmarkResult]:
    rel_path = TOUCH_SCENARIOS[scenario]
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {scenario} ({touch_mode}) -- touch {rel_path}")
    print(f"{'=' * 60}")

    results = {t.name: BenchmarkResult(scenario=f"{scenario}-{touch_mode}", tool=t.name) for t in tools}

    # Warm caches
    print("  Warming caches...")
    for tool in tools:
        tool.build(str(fixture_copy))

    file_path = fixture_copy / rel_path
    pristine = file_path.read_text()

    for run_idx in range(1, runs + 1):
        print(f"\n  Run {run_idx}/{runs}:")
        for tool in tools:
            # Restore the touched file to its pristine state. Faster
            # than rmtree+copytree (which copied ~hundreds of MB) and
            # sufficient because we only ever touch one file per run.
            file_path.write_text(pristine)

            # Apply touch
            touch_file(file_path, touch_mode)

            elapsed, built = tool.build(str(fixture_copy))
            results[tool.name].times.append(elapsed)
            results[tool.name].builds.append(built)
            print(f"    [{tool.name}] {elapsed:.2f}s -- {built} drvs built")

    # Always restore.
    file_path.write_text(pristine)
    return [results[t.name] for t in tools]


def run_no_change_benchmark(
    tools: list[NixTool],
    fixture_copy: Path,
    runs: int,
) -> list[BenchmarkResult]:
    print(f"\n{'=' * 60}")
    print("SCENARIO: no-change (cache validation overhead)")
    print(f"{'=' * 60}")

    results = {t.name: BenchmarkResult(scenario="no_change", tool=t.name) for t in tools}

    print("  Warming caches...")
    for tool in tools:
        tool.build(str(fixture_copy))

    for run_idx in range(1, runs + 1):
        print(f"\n  Run {run_idx}/{runs}:")
        for tool in tools:
            elapsed, built = tool.build(str(fixture_copy))
            results[tool.name].times.append(elapsed)
            results[tool.name].builds.append(built)
            print(f"    [{tool.name}] {elapsed:.2f}s -- {built} drvs built")

    return [results[t.name] for t in tools]


def _significant(winner: BenchmarkResult, runner_up: BenchmarkResult) -> bool:
    """Two means are 'significantly different' if their 1σ bands don't
    overlap. Loose by hyperfine standards but enough to flag noise."""
    if not winner.times or not runner_up.times:
        return False
    return (winner.mean + winner.stddev) < (runner_up.mean - runner_up.stddev)


def format_results(all_results: list[list[BenchmarkResult]]) -> str:
    lines = [f"\n{'=' * 70}", "BENCHMARK RESULTS SUMMARY", "=" * 70]
    if not all_results or not all_results[0]:
        return "\n".join(lines + ["(no results)"])

    tools = [r.tool for r in all_results[0]]
    # Per-tool: (mean wall, mean drvs built)
    headers = ["Scenario"] + [f"{t} (s / drvs)" for t in tools] + ["Winner", "Speedup"]
    header = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("-" * (len(c) + 2) for c in header.split("|")[1:-1]) + "|"
    lines += ["\n" + header, sep]

    for scenario_results in all_results:
        by_tool = {r.tool: r for r in scenario_results}
        scenario_name = scenario_results[0].scenario
        winner_name = min(by_tool, key=lambda k: by_tool[k].mean)
        winner = by_tool[winner_name]
        # Pick the second-fastest by mean — if it's not significantly
        # slower, the "winner" label is misleading.
        others = sorted(
            (r for r in scenario_results if r.tool != winner_name),
            key=lambda r: r.mean,
        )
        if others and _significant(winner, others[0]):
            winner_label = f"**{winner_name}**"
            runner_up = others[0]
            speedup = f"{runner_up.mean / winner.mean:.2f}x" if winner.mean > 0 else "--"
        elif others:
            winner_label = "tie"
            speedup = "n.s."
        else:
            winner_label = f"**{winner_name}**"
            speedup = "--"
        cells = " | ".join(
            f"{by_tool[t].mean:.2f} / {by_tool[t].builds_mean:.0f}" for t in tools
        )
        lines.append(f"| {scenario_name} | {cells} | {winner_label} | {speedup} |")

    lines.append("\n## Detailed Results\n")
    for scenario_results in all_results:
        scenario_name = scenario_results[0].scenario
        lines.append(f"### {scenario_name}\n")
        for r in scenario_results:
            lines.append(f"**{r.tool}:**")
            lines.append(f"  - Wall: {r.mean:.2f}s (+/-{r.stddev:.2f}s) range {r.min:.2f}s..{r.max:.2f}s")
            lines.append(f"  - Drvs built: {r.builds_mean:.1f} (per-run: {r.builds})")
            lines.append("")
    return "\n".join(lines)


def export_json(all_results: list[list[BenchmarkResult]], output_path: Path) -> None:
    data = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": [],
    }
    for scenario_results in all_results:
        entry: dict[str, object] = {"name": scenario_results[0].scenario}
        for r in scenario_results:
            entry[r.tool] = {
                "times": r.times,
                "mean": r.mean,
                "stddev": r.stddev,
                "builds": r.builds,
                "builds_mean": r.builds_mean,
            }
        data["scenarios"].append(entry)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults exported to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark incremental builds for go2nix modes."
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--scenario",
        choices=["no_change", *TOUCH_SCENARIOS.keys(), "all"],
        default="all",
    )
    parser.add_argument(
        "--touch-mode",
        choices=list(_TOUCH_TEMPLATES.keys()),
        default="private",
        help="Edit type: private=internal symbol, exported=API change (default: private)",
    )
    parser.add_argument(
        "--tools",
        default="nix,nix-ca",
        help="Comma-separated tools (default: nix,nix-ca)",
    )
    parser.add_argument("--json", type=Path, help="Export results as JSON")
    parser.add_argument(
        "--assert-cascade",
        type=int,
        default=None,
        metavar="N",
        help="Fail if any tool builds more than N derivations on a touch "
        "scenario. Use as a regression check for the iface cutoff.",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    fixture_src = repo_root / "tests" / "fixtures" / "torture-project"
    go2nix_src = str(repo_root)

    print("Resolving dependencies...")
    nixpkgs_path, plugin_path, gomodcache = resolve_paths(repo_root)

    # Detect system
    _, system, _ = run_command(["nix", "eval", "--raw", "--impure", "--expr", "builtins.currentSystem"])
    system = system.strip()

    # Write nix expressions to a temp dir
    tmpdir = Path(os.environ.get("TMPDIR", "/tmp")) / "bench-incremental"
    tmpdir.mkdir(parents=True, exist_ok=True)

    # All builds use a local socat daemon with ca-derivations enabled.
    # This ensures a fair comparison: same store, same sandbox=false,
    # same nix binary with the plugin. Dependencies are fetched from the
    # system daemon store.
    local_daemon = LocalDaemon(tmpdir, "ca-derivations")
    local_daemon.start()
    # Disable network substituters: every CA-mode build would otherwise
    # query cache.nixos.org for the realisation of every local CA
    # derivation (~1s of HTTPS round-trips per run, all 404s — local
    # CA outputs are by definition not in any public cache). The
    # `daemon` substituter still serves third-party packages from the
    # system store. allowSubstitutes=false on the per-package drvs (set
    # by caAttrs in dag/default.nix) is the structural fix; this guard
    # ensures the bench numbers don't depend on it being set correctly.
    common_opts = [
        "--option", "sandbox", "false",
        "--option", "substituters", "daemon",
    ]

    expr_nix = write_nix_expr(tmpdir, "nix", str(fixture_src), go2nix_src, system)
    expr_ca = write_nix_expr(tmpdir, "nix-ca", str(fixture_src), go2nix_src, system, "contentAddressed = true;")

    available_tools: dict[str, NixTool] = {
        "nix": NixTool("nix", nixpkgs_path, plugin_path, gomodcache, expr_nix,
                        common_opts, local_daemon),
        "nix-ca": NixTool("nix-ca", nixpkgs_path, plugin_path, gomodcache, expr_ca,
                           common_opts + ["--option", "extra-experimental-features", "ca-derivations"],
                           local_daemon),
    }

    requested = [t.strip() for t in args.tools.split(",")]
    tools: list[NixTool] = []
    for name in requested:
        if name in available_tools:
            tools.append(available_tools[name])
        else:
            parser.error(f"Unknown tool: {name!r} (available: {list(available_tools)})")

    # Copy fixture (one-time; we restore touched files in-place per run).
    fixture_copy = Path(os.environ.get("TMPDIR", "/tmp")) / "bench-fixture-copy"
    if fixture_copy.exists():
        shutil.rmtree(fixture_copy)
    shutil.copytree(fixture_src, fixture_copy)

    # Sanity check: ensure each requested tool's expression instantiates
    # to the kind of derivation it claims. nix-ca should produce a CA
    # derivation closure; if `contentAddressed = true` was silently
    # ignored, we'd benchmark two identical configs and get a meaningless
    # tie. Cheap eval-only check, no build.
    for tool in tools:
        if "ca" in tool.name:
            cmd = [
                "nix-instantiate", "-I", f"nixpkgs={nixpkgs_path}",
                "--option", "plugin-files", plugin_path,
                "--option", "allow-import-from-derivation", "true",
                "--option", "extra-experimental-features", "ca-derivations",
                "--eval", "--json", "--expr",
                f"let drv = import {tool.expr_path} {{}}; "
                f"in builtins.any (x: x ? __contentAddressed) "
                f"  (builtins.attrValues drv.passthru.localPackages or {{}})"
                f" || (drv.drvAttrs ? __contentAddressed)",
            ]
            _, stdout, _ = run_command(cmd)
            if stdout.strip() not in ("true", "false"):
                # Eval failed (likely passthru attr name differs); fall
                # through to a build-time check.
                continue
            if stdout.strip() == "false":
                print(f"  WARNING: {tool.name} expression has no CA derivations "
                      f"in its closure — contentAddressed may be silently ignored")

    print(f"\n{'=' * 70}")
    print("GO2NIX INCREMENTAL BUILD BENCHMARK")
    print(f"{'=' * 70}")
    print(f"Fixture:    torture-project/app-full")
    print(f"Tools:      {', '.join(t.name for t in tools)}")
    print(f"Touch mode: {args.touch_mode}")
    print(f"Runs:       {args.runs}")
    print(f"Store:      {local_daemon.store_root}")

    scenarios = (
        ["no_change", *TOUCH_SCENARIOS.keys()]
        if args.scenario == "all"
        else [args.scenario]
    )

    all_results: list[list[BenchmarkResult]] = []
    for name in scenarios:
        if name == "no_change":
            all_results.append(run_no_change_benchmark(tools, fixture_copy, args.runs))
        else:
            all_results.append(
                run_touch_benchmark(
                    tools, fixture_copy, name, args.touch_mode, args.runs
                )
            )

    print(format_results(all_results))

    if args.json:
        export_json(all_results, args.json)

    # Cleanup before asserting so a failure doesn't leak the daemon.
    local_daemon.stop()
    shutil.rmtree(fixture_copy, ignore_errors=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Cascade-size regression check: fail loudly if any tool's
    # worst-case build count on a touch scenario exceeds the threshold.
    # Useful as a CI guardrail for the iface cutoff.
    if args.assert_cascade is not None:
        violations = []
        for scenario_results in all_results:
            for r in scenario_results:
                if r.scenario == "no_change":
                    continue
                if r.builds_max > args.assert_cascade:
                    violations.append(
                        f"{r.scenario}/{r.tool}: built {r.builds_max} drvs "
                        f"(threshold {args.assert_cascade})"
                    )
        if violations:
            print(f"\n{'=' * 70}")
            print(f"FAIL: cascade-size threshold ({args.assert_cascade}) exceeded")
            print(f"{'=' * 70}")
            for v in violations:
                print(f"  {v}")
            raise SystemExit(1)
        print(f"\nPASS: all tools stayed within cascade threshold {args.assert_cascade}")


if __name__ == "__main__":
    main()
