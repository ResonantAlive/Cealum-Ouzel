# Public source bundle contents

This directory is intentionally a source-only release candidate.

Included:

- model, packed-data, pre-training, CPT, SFT, DPO, tokenizer, and dashboard code;
- environment checks, benchmarks, and tests;
- Apache-2.0 licensing, dependency lists, configuration, and example manifests.

Excluded:

- model weights, optimizer states, checkpoints, and experiment logs;
- tokenizer artifacts, raw corpora, encoded shards, validation data, and sample outputs;
- server addresses, credentials, local paths, virtual environments, caches, and temporary files.

Before publishing, run the environment gate and tests on a clean checkout, then
review the model card and data-provenance claims for the corresponding release.
