# Portability

This experiment directory is self-contained:

- Python entrypoints live at the repo root of this folder
- Baseline package is local: `src/`
- Default inputs/outputs are under `data/` and `anonymized_results/`

Copy this whole directory to another machine, set `DEEPSEEK_API_KEY` (or pass `--api-key`), then run e.g.:

```bash
python zxz_synthpai_deepseek_anonymize.py --dry-run --num-profiles 1
python zxz_synthpai_deepseek_causal_anonymize.py --dry-run --num-profiles 1
```
