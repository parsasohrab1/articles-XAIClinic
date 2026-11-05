# articles-XAIClinic

How stable are feature explanations of clinical classifiers when the training data are resampled?
Real data: Wisconsin Diagnostic Breast Cancer (UCI, n = 569, 30 features). TRL 4 (public real data, laboratory validation only).

```
pip install -e ".[dev]" && python -m pytest
# data: https://archive.ics.uci.edu/dataset/17 (not committed)
python research/run_study.py /path/to/wdbc.data
```
