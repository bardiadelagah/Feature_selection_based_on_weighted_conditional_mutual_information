from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC

class modelClassifier:
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    y_predict_knn = None
    y_predict_svm = None
    svm_model = None
    knn_model = None

    def __init__(self, New_data_discretized, label_num):
        x = New_data_discretized
        y = label_num
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    def fit_KNN_model(self):
        # train an KNN classifier
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(self.x_train, self.y_train)
        self.knn_model = neigh
        self.y_predict_knn = neigh.predict(self.x_test)
        print("metrics for KNN model")
        self.compute_metrics(self.y_test, self.y_predict_knn)

    def fit_SVM_model(self):
        # train a linear SVM classifier
        clf = LinearSVC()
        clf.fit(self.x_train, self.y_train)
        self.svm_model = clf
        self.y_predict_svm = clf.predict(self.x_test)
        print("metrics for Linear SVM model")
        self.compute_metrics(self.y_test, self.y_predict_svm)

    def compute_metrics(self, y_true, y_pred):
        # compute metrics
        precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        print("precision")
        print(precision)
        print("recall")
        print(recall)
        print("f_score")
        print(f_score)
        print("y_true")
        print(y_true)
        print("y_pred")
        print(y_pred)
