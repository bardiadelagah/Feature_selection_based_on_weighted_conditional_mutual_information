from sonarClass import sonarClass
from WCFRclass import WCFRclass
from modelClassifier import modelClassifier

sonar_class = sonarClass()
wcrf_class = WCFRclass(sonar_class.data, sonar_class.data_discretized, sonar_class.label_num)

mode = ['run_WCRF','train_models']
mode = mode[1]

if mode == 'run_WCRF':
    wcrf_class.WCRFalgoritmForSonarData()
else:
    wcrf_class.setSmanual(
        [11, 26, 20, 35, 30, 10, 18, 44, 34, 19, 36, 9, 43, 12, 31, 45, 42, 8, 21, 39, 13, 33, 32, 40, 46,
         41, 14, 23, 47, 7, 28, 5, 6, 27, 4, 38, 3, 22, 1, 2, 0, 48, 49, 50, 51, 15, 52, 53, 54, 55])

    # make feature vector by S array(50 candidate features)
    wcrf_class.setNewDataDiscretized()

    # fit and predict by SVM and KNN classifiers
    modelClassifier_instance = modelClassifier(wcrf_class.New_Data_Discretized, wcrf_class.label_num)
    modelClassifier_instance.fit_KNN_model()
    print("=======")
    modelClassifier_instance.fit_SVM_model()
