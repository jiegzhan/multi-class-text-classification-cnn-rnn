### Project: Classify Kaggle San Francisco Crime Description

### Highlights
- Multi-class text classification (sentence classification) problem.
- Classify Kaggle San Francisco Crime **Descript** into 39 **Category** labels.
- Hybrid **TextCNN + GRU** model implemented with **TensorFlow 2.x Keras**.

### Data: [Kaggle San Francisco Crime](https://www.kaggle.com/c/sf-crime/data)
- Input: **Descript**
- Output: **Category**

Examples:

| Descript | Category |
| --- | --- |
| GRAND THEFT FROM LOCKED AUTO | LARCENY/THEFT |
| POSSESSION OF NARCOTICS PARAPHERNALIA | DRUG/NARCOTIC |

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training data is included at `./data/train.csv.zip`. Prediction sample data is at `./data/small_samples.csv`.

### Train
```bash
python3 train.py ./data/train.csv.zip ./training_config.json
```

Artifacts are written to `./trained_results_<timestamp>/`:
- `saved_model/` — exported SavedModel for inference
- `best_model.keras` — best validation checkpoint (primary load path for predict)
- `words_index.json` — vocabulary mapping
- `labels.json` — class labels
- `trained_parameters.json` — hyperparameters and sequence length

### Predict
```bash
python3 predict.py ./trained_results_<timestamp>/ ./data/small_samples.csv
```

Predictions are saved to `./predicted_results_<timestamp>/predictions_all.csv`.

### Reference
- [Implement a CNN for text classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
