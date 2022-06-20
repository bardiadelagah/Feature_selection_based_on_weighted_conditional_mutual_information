# Feature selection based on weighted conditional mutual information

This code is a python implementation for "Feature selection based on weighted conditional mutual information" article. Link of article is here [<a href="https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Feature+selection+based+on+weighted+conditional+mutual+information&btnG=" target="_blank">Google scholar</a>]
The video discription of code in persian language is here [<a href="https://www.youtube.com/" target="_blank">Youtube</a>]


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

### modelClassifier

This class get data after disciritisize by sonarClass and fit a Linear SVM and a KNN model. Then compute metrics for fitted models.

### WCFRclass

This class implement the algorithem of article. The class constroctor get data, data_discretized and the lable_num that made by sonarClass from orginal dataset.
Then find best features base on algorithem in article.

## How it works

Run main.py for find best features on sonar dataset. beacuse of high time-consuming for run the code, we run code one time and find best 50 featurs from all featurs in sonar dataset. The column number of features are in a list 

```python
wcrf_class.setSmanual(
        [11, 26, 20, 35, 30, 10, 18, 44, 34, 19, 36, 9, 43, 12, 31, 45, 42, 8, 21, 39, 13, 33, 32, 40, 46,
         41, 14, 23, 47, 7, 28, 5, 6, 27, 4, 38, 3, 22, 1, 2, 0, 48, 49, 50, 51, 15, 52, 53, 54, 55])
```





## Donate us
If you like our project and it's useful, feel free to donate us.

Bitcoin(BTC): bc1qs2fatdfdvc5jyq4a0f5t7plmy8sxmyk08tq5e5

Ethereum(ETH): 0x5847D46Bfed82a475ef4187cfBD55EF412C05093

Tether(USDT-TRC20): TAmbZwJXDZ8bo2hjGXtNkTSEYi8dt2Xww8

XRP(XRP): rqTpCtGtBEhcPjZLXfNTv3JbCdkRKGLCF

Dogecoin(DOGE): DGZYMS6nnT3cBYwDtSD7VVubr1dSfykURC

TRON(TRX): TAmbZwJXDZ8bo2hjGXtNkTSEYi8dt2Xww8

BitTorrent-New(BTT-BEP20): 0x5847D46Bfed82a475ef4187cfBD55EF412C05093

Decentraland(MANA-ERC20): 0x5847D46Bfed82a475ef4187cfBD55EF412C05093
