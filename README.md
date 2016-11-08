### Project: Classify Kaggle San Francisco Crime Description

### Highlights:
  - The goal of this project is to classify Kaggle San Francisco Crime Description into 39 classes.
  - This model was built with CNN, RNN (GRU and LSTM) and word embeddings on Tensorflow.

### Data: [Kaggle San Francisco Crime](https://www.kaggle.com/c/sf-crime/data)
  - Input: Descript
  - Output: Category
  - Examples:

    Descript   | Category
    -----------|-----------
    GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT
    POSSESSION OF NARCOTICS PARAPHERNALIA|DRUG/NARCOTIC
    AIDED CASE, MENTAL DISTURBED|NON-CRIMINAL
    AGGRAVATED ASSAULT WITH BODILY FORCE|ASSAULT
    ATTEMPTED ROBBERY ON THE STREET WITH A GUN|ROBBERY
    
  > **Note**
  > - 39 classes are: ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION', 'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']
    
### Train:
  - Command: ```python3 train.py train_data.file train_parameters.file```
  - Example: ```python3 train.py ./data/train.csv.zip ./training_config.json```

### Predict:
  - Command: ```python3 predict.py ./trained_results_dir/ test_file.csv```
  - Example: ```python3 predict.py ./trained_results_1478563595/ ./data/small_samples.csv```
