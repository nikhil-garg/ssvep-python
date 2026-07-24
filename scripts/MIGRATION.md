# Script migration

Supported entry points are being consolidated behind `ssvep` commands and
YAML studies. Historical R&F analyses remain available under `scripts/legacy/`.

| Previous purpose | Supported replacement |
| --- | --- |
| Preprocessing | `ssvep preprocess --config <config>` |
| Dataset inspection | `ssvep inspect --data-dir <dataset>` |
| YAML study planning | `ssvep study run --config configs/studies/<study>.yaml` |
| Dashboard rendering | `ssvep dashboard` |

Legacy scripts retain their original numerical implementation and outputs.
They are not used by the supported launcher path.
