
# All plots used in the submission are directly from the script
python src/anonymized/plot_anonymized.py --type base --out_path plots_pdf --attributes fair --settings no-base
python src/anonymized/plot_anonymized.py --type base --out_path plots_pdf --attributes all --settings no-base
python src/anonymized/plot_anonymized.py --type base --out_path plots_pdf --attributes fair --settings all 
python src/anonymized/plot_anonymized.py --type base --out_path plots_pdf --attributes all --settings all 
python src/anonymized/plot_anonymized.py --type base --out_path plots_pdf --attributes fair --settings no-base iter=3
python src/anonymized/plot_anonymized.py --type base --out_path plots_pdf --attributes all --settings no-base iter=3
python src/anonymized/plot_anonymized.py --type base --out_path plots_pdf --attributes fair --settings all iter=3
python src/anonymized/plot_anonymized.py --type base --out_path plots_pdf --attributes all --settings all iter=3
python src/anonymized/plot_anonymized.py --type base --out_path plots_pdf --attributes all --settings base_ablation
python src/anonymized/plot_anonymized.py --type base --out_path plots_pdf --attributes all --settings main_base_ablation

python src/anonymized/plot_anonymized.py --type synthetic --out_path plots_pdf --attributes fair --settings no-base
python src/anonymized/plot_anonymized.py --type synthetic --out_path plots_pdf --attributes all --settings no-base
python src/anonymized/plot_anonymized.py --type synthetic --out_path plots_pdf --attributes fair --settings all 
python src/anonymized/plot_anonymized.py --type synthetic --out_path plots_pdf --attributes all --settings all 
python src/anonymized/plot_anonymized.py --type synthetic --out_path plots_pdf --attributes all --settings base_ablation