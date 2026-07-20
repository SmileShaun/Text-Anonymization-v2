# Portability

This experiment directory is self-contained:

- Python entrypoints live at the repo root of this folder
- Baseline package is local: `src/`
- Dataset builder raw file: `data/synthpai/synthpai.jsonl`
- Prepared inputs under `data/inputs/` and `data/base_inferences/`

Copy this whole directory to another machine, set `DEEPSEEK_API_KEY` (or pass `--api-key`), then run e.g.:

```bash
python zxz_synthpai_deepseek_anonymize_full_comments.py --dry-run --num-profiles 1
python zxz_build_synthpai_input_datasets.py
```
