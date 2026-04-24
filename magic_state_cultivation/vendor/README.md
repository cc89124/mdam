# Vendored Code

This directory contains circuit generation and decoder code from the
magic state cultivation paper by Gidney, Shutty, and Jones.

**Source:** https://github.com/Strilanc/magic-state-cultivation
**Commit:** `871e68ff6df2f75190b1bfd6351459d1b5a037e3` (main branch)
**Paper:** "Magic state cultivation: growing T states as cheap as CNOT gates" ([arXiv:2409.17595](https://arxiv.org/abs/2409.17595))

## Contents

- `src/` — Python library for circuit construction (`cultiv/`) and
  general QEC utilities (`gen/`). Includes the `CompiledDesaturationSampler`
  decoder used for Y_L decoding and gap confidence.
- `tools/` — CLI tools for circuit generation (`make_circuits`, etc.).

## Usage

Our scripts add `vendor/src` to `sys.path` so that `from src.cultiv import ...`
and `import gen` work without installation.
