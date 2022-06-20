# Feature selection based on weighted conditional mutual information

This code is a python implementation for "Feature selection based on weighted conditional mutual information" article. Link of article is here [<a href="https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Feature+selection+based+on+weighted+conditional+mutual+information&btnG=" target="_blank">Google scholar</a>]


# Structure

The folder and files are below:

```bash
├── main.py                   # for run the code
├── modelClassifier.py        # fit Linear SVM and KNN classifiers
├── requirements.txt          # requirements and dependensies
├── sonar.all-data            # Dataset
├── sonarClass.py             # preproscceing on sonal.all-data file
└── WCFRclass.py              # Implementation the alghoritem of article
```

### sonarClass

This class preprocessing on data. It get data and change string lables to unique number. Then it discirtisize all data base on method exist in article.
