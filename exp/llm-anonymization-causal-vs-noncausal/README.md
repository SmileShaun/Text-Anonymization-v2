# Overview

This is the repository accompanying our ICLR 2025 paper ["Large Language Models are Advanced Anonymizers"](https://arxiv.org/abs/2402.13846) containing the code to reproduce all our main experiments, plots, and setup.

## Setup

Install mamba via:

```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

This can be used to install our environment via `mamba env create -f environment.yaml`

Afterward, you should edit the file `credentials_clean.py` to contain your OpenAI and Azure keys, respectively. Then rename it to `credentials.py` so that it will be loaded ( and is contained in the `.gitignore`).

Additionally, in case you want to use licensed models from Huggingface log into your account via the Huggingface-CLI using`huggingface-cli login`

## Important notes before running

Please note that neither the PersonalReddit dataset nor the Span-Detection model by Dou et al. are open-access. Therefore, both have been removed from this repository. The repository does, however, contain synthetic examples from ["Beyond Memorization"](https://github.com/eth-sri/llmprivacy). Corresponding configuration files can be found under `configs/anonymization/synthetic`.

You can also use this synthetic dataset to evaluate personal attribute inferences capabilities: [A Synthetic Dataset for Personal Attribute Inference](https://github.com/eth-sri/SynthPAI). It consists a large-scale fully-synthetic dataset as well as a data generation pipelline. In our corresponding [paper](https://arxiv.org/abs/2406.07217) we show that the dataset is a good proxy for real-world data, allowing all the same conclusions across all experiments, and can be used to evaluate personal attribute inference in a privacy-preserving manner. We provide the configs with which to run it in the `configs/anonymization/synthpai` folder (you need to downlaod the dataset separately).

## Running

We provide a wide range of configurations to run our experiments. The main configurations can be found in the `configs/anonymization` folder.

Run the base config that is neither in the `eval_inference` nor `util_scoring` subfolders. For example `configs/anonymization/synthetic/reddit_gpt4.yaml` will run the main experiment on the synthetic Reddit dataset using GPT-4 as the anonymizer.

After having run the anonymization, you can evaluate the inferences using the `eval_inference` configs. We generally use GPT-4 as the judge for these inferences. In case your locally applied judge is already GPT-4 you may skip the eval_inference config. Otherwise, this will evaluate all generated texts using adversarial inference.

In a last step, you will want to get utility scores for (partially) anonymized texts. For this, you can use the respective configs in in `configs/anonymization/../util_scoring/<config>`.

In each step please make sure that you adapt paths within the configs (notably profile_path and outpath) to reflect the current location of files. (Side note: You will find that as a cost-saving measure, we shared the inferences on fully non-anonymized text).

Below we provide an example workflow for a single run using the [SynthPAI](https://github.com/eth-sri/SynthPAI) dataset:

```bash
# SynthPAI Inference
python main.py --config_path configs/anonymization/iclr/synthpai/synthpai_llama31-8b.yaml

# Optional - Use this merge script if you dropped e.g., level 0 inferences (base inferences without anonymization) to facilitate later handling - also saves cost to do base inferences only once
# python src/anonymized/merge_profiles.py --in_paths anonymized_results/iclr_synthpai/llama31-8b/inference_3.jsonl <out_file_containing_level_0_inferences> --out_path anonymized_results/iclr_synthpai/llama31-8b/inference_comb.jsonl

# Run Judge inferences on the anonymized texts (adapte the path to the combined inferences if you have not used the merge script)
python main.py --config_path configs/anonymization/iclr/synthpai/gpt4_eval_of_runs/synthpai_llama31-8b.yaml

# Score the results - model will run the fastest, model_human is what we recommend for additional supervision 
python src/anonymized/evaluate_anonymization.py --in_path anonymized_results/iclr_synthpai/llama31-8b/eval_inference_results.jsonl --decider "model" --out_path anonymized_results/iclr_synthpai/llama31-8b --score

# Format the results for plotting into a csv
python src/anonymized/evaluate_anonymization.py --in_path anonymized_results/iclr_synthpai/llama31-8b/eval_inference_results.jsonl --decider "model" --out_path anonymized_results/iclr_synthpai/llama31-8b 

# From here you can plot the results using the plotting scripts
```

## Evaluation Explanation

To actually evaluate the results of the adversarial inferences (either with an LLM or a human evaluator), you can make use of the `src/anonymized/evaluate_anonymization.py` script.

In particular, you want to run it as follows:

`python src/anonymized/evaluate_anonymization.py --in_path <your_eval_inference_with_utility>.jsonl --decider "model_human" --out_path <out_directory> --score`

to create a canonical `eval_out.jsonl`. Running the same command again without the `--score` will translate this into the csv format used in our plotting script.

## Plotting

All our plots have been created with a single call to the `all_plots.sh` script. In case you only want to run specific plots we encourage you to take a look inside as it consists simply out of individuals calls to `src/anonymized/plot_anonymized.py`.

## Citation

If you use this code, please consider citing our work:

```bibtex
@inproceedings{
    staab25lmanon,
    title={Language Models are Advanced Anonymizers},
    author={Robin Staab and Mark Vero and Mislav Balunović and Martin Vechev},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
}
```
