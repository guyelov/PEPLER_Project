# PEPLER (PErsonalized Prompt Learning for Explainable Recommendation)

## Paper
- Lei Li, Yongfeng Zhang, Li Chen. [Personalized Prompt Learning for Explainable Recommendation](https://arxiv.org/abs/2202.07371). ACM Transactions on Information Systems (TOIS), 2023.

## Datasets to [download](https://lifehkbueduhk-my.sharepoint.com/:f:/g/personal/16484134_life_hkbu_edu_hk/Eln600lqZdVBslRwNcAJL5cBarq6Mt8WzDKpkq1YCqQjfQ?e=cISb1C)
- TripAdvisor Hong Kong
- Amazon Movies & TV
- Yelp 2019

For those who are interested in how to obtain (feature, opinion, template, sentiment) quadruples, please refer to [Sentires-Guide](https://github.com/lileipisces/Sentires-Guide).

## Usage
Below are examples of how to run PEPLER for reproducing the results and when using our proposed improvement method.
The main branch contains the code for reproducing the results.
In addition, the config file contains more parameters, which are the default parameters as used in the paper.

To run our improvement method, simply switch to the selector_fixed branch and run one of these commands:
```
python -u main.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--checkpoint ./tripadvisor/ >> tripadvisor.log

python -u discrete.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--checkpoint ./tripadvisord/ >> tripadvisord.log

python -u reg.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--use_mf \
--checkpoint ./tripadvisormf/ >> tripadvisormf.log

python -u reg.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--rating_reg 1 \
--checkpoint ./tripadvisormlp/ >> tripadvisormlp.log
```

## Code dependencies
- Python 3.6
- PyTorch 1.6
- transformers 4.18.0

## Code reference
- [mkultra: Prompt Tuning Toolkit for GPT-2](https://github.com/corolla-johnson/mkultra)
- [PEPLER: PErsonalized Prompt Learning for Explainable Recommendation](https://github.com/lileipisces/PEPLER?tab=readme-ov-file)
## Citation
```
@article{TOIS23-PEPLER,
	title={Personalized Prompt Learning for Explainable Recommendation},
	author={Li, Lei and Zhang, Yongfeng and Chen, Li},
	journal={ACM Transactions on Information Systems (TOIS)},
	year={2023}
}
```
