# blis-sglang

A Go model of the SGLang inference engine's scheduling, KV-cache, hierarchical-cache,
disaggregation and metrics behaviour, for the BLIS simulator. It implements the engine-neutral
contract in [blis-substrate](https://github.com/inference-sim/blis-substrate) `contracts/`
(`engine.Engine`), so a simulator such as
[inference-sim](https://github.com/inference-sim/inference-sim) builds it once and holds only that
interface. The design, which states every rule below and why, is at
<https://inference-sim.github.io/blis-design/>.

## What it owns, and what it never holds

It owns everything about SGLang's behaviour: defaults, how a batch is formed, which KV pages move,
which kernel runs, and how a server-argument set resolves. It never holds the cost model, a
hardware fact, a coefficient or a router.

- **One release per build.** A build models one SGLang release, named in `provenance.go`. Every
  statement about SGLang cites `sglang/<path>:<lines>@<commit8>` at that release.
- **Time is a value.** Nothing reads a clock or starts a goroutine; the simulator passes the
  simulated instant into every call.
- **Checked against SGLang itself.** CI runs the release's own Python classes on CPU, with only
  the GPU, the clock, transports and the platform replaced (`oracle/`), and compares every
  decision with this module's.
- **Every server argument and environment variable has a status** (`fields.json`, from
  `fields/` and the release's source), and every connector, policy and cache type has one
  (`surface.json`, from `tools/surface/`). Both are generated; CI checks they are current.

The engine port is not written yet. This repository holds its skeleton: the release pin, the
shared checks from blis-substrate's `contracts/harness`, the field and surface lists, the metrics
manifest (`oracle/metrics/`) and CI.

## Connections

- **Imports** only blis-substrate's `contracts/` packages.
- **Is pinned by** inference-sim, which builds the engine at one site and holds only
  `engine.Engine`.
- **Is read by** accuracy-dashboard and blis-config-search through the field list and the surface
  list.

## Build, test and run

```sh
go build ./... && go vet ./... && go test ./...
```

`.github/workflows/ci.yml` and `.github/workflows/ci-light.yml` hold the job lists and say which
changes select which jobs. `AGENTS.md` holds the rules an AI-assisted change follows.
