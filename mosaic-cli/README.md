# mosaic-cli

A small command-line interface for inspecting, creating, and exercising Mosaic embedding tables. It exists primarily as **debug / introspection tooling**, not as a production interface — every command is a thin shell around `mosaic-core`'s public API.

> **Constraint:** the CLI is forbidden from touching any `internal` symbol in `mosaic-core`. Kotlin enforces this at the module boundary (mosaic-cli is a separate Gradle module, so `internal` is unreachable). The acceptance criterion "CLI apenas usa API pública" is satisfied by construction.

## Running

During development:

```bash
./gradlew :mosaic-cli:run --args="<command> <args...>"
```

To produce a distributable binary:

```bash
./gradlew :mosaic-cli:installDist
./mosaic-cli/build/install/mosaic-cli/bin/mosaic-cli --help
```

Exit codes:

| Code | Meaning                                                       |
|-----:|---------------------------------------------------------------|
| 0    | Success                                                        |
| 1    | Usage error — missing/invalid arguments, unknown command       |
| 2    | Runtime error — file not found, corrupted file, checksum bad   |

## Commands

### `create` — build and save a new table

```bash
mosaic-cli create \
  --vocab-size 50000 \
  --dim 128 \
  --initializer uniform \
  --seed 42 \
  --output embeddings.bin
```

| Flag                  | Required | Default        | Meaning                                       |
|-----------------------|:--------:|----------------|-----------------------------------------------|
| `--vocab-size N`      | ✓        |                | Number of token rows                           |
| `--dim D`             | ✓        |                | Embedding dimension                            |
| `--output / -o PATH`  | ✓        |                | Output `.bin` path; sidecar written next to it |
| `--initializer NAME`  |          | `uniform`      | `uniform`, `xavier`, `he`, `zeros`, `constant` |
| `--seed N`            |          | `42`           | Seed for random initializers                   |
| `--bound F`           |          | `0.5/dim`      | Bound for the `uniform` initializer            |
| `--value F`           | for `constant` | —        | Constant value (only with `--initializer constant`) |

### `inspect` — show metadata + checksum status

```bash
mosaic-cli inspect --input embeddings.bin
```

Sample output:

```text
Mosaic Embedding Table
─────────────────────────────────
Version:             1
Vocab size:          50,000
Embedding dim:       128
Format:              float32-le
File size:           24.41 MB
Checksum:            a3f4b2e1d... (valid ✓)
Created:             2026-05-21T13:13:26.775Z
Tessera-compatible:  yes
```

Exits `2` if the checksum recorded in `.meta.json` doesn't match the SHA-256 of the `.bin` file.

### `stats` — distribution statistics

```bash
mosaic-cli stats --input embeddings.bin
```

Computes min / max / mean / stddev of every float in the matrix, plus min / max / mean of the per-row L2 norms. Loads the table once into memory.

### `similar` — top-K nearest neighbors

By token ID:

```bash
mosaic-cli similar --embeddings embeddings.bin --id 1234 --top-k 10
```

By text (uses Tessera tokenizer to encode, then mean-pools the resulting vectors):

```bash
mosaic-cli similar \
  --embeddings embeddings.bin \
  --tokenizer tessera.json \
  --text "gato" \
  --top-k 10
```

`--id` and `--text` are mutually exclusive. `--text` requires `--tokenizer`. `--top-k` defaults to 10.

### `encode` — text → vectors

```bash
mosaic-cli encode \
  --tokenizer tessera.json \
  --embeddings embeddings.bin \
  --text "Olá mundo" \
  --format json
```

| `--format` value | Output                                                 |
|------------------|--------------------------------------------------------|
| `pretty` (default) | One line per token: `token NNN  [v1, v2, v3, v4, ...]` |
| `json`             | A single JSON object with `ids` array + `vectors` array |
| `csv`              | One row per token: `id,v1,v2,...,vDim`                  |

## Tests

CLI integration tests live in `src/test/kotlin/dev/mosaic/cli/`. They use temp directories, train a small Tessera tokenizer inline, and run every command end-to-end via the testable `runCli(args)` entry point.

```bash
./gradlew :mosaic-cli:test
```
