import numpy as np


class sonarClass:
    data = np.array([])
    label_original = np.array([])
    label_num = np.array([])
    data_normalized = np.array([])
    data_discretized = np.array([])

    def __init__(self):
        [self.data, self.label_original] = self.sonarDatareadFile()
        self.label_num = self.sonarLabelToNum(self.label_original)
        self.data_normalized = self.sonarDataNormalize(self.data)
        self.data_discretized = self.sonarDataDiscretized(self.data_normalized)

    def sonarDatareadFile(self):
        '''
            Sonar data set file has 208 lines. each line has 60 feature + 1 label for classification
            data used for manage features. shape is 208 rows and 60 columns
            label used for classification. shape is 208 rows and 1 column
            this function first read file then give them to data an label nd-array, and change data nd-array form string
            to float
            :returns data (in shape 208 * 60) and label(in shape 208*1) as nd-array
        '''
        data = np.array([])
        label = np.array([])
        with open("sonar.all-data", "r") as a_file:
            for line in a_file:
                stripped_line = line.strip()
                my_list = stripped_line.split(',')  # split each len by ',' for separate numbers and label
                my_list_size = len(my_list)  # my_list_size equal to 61 (index start from 0 to 60)
                data = np.append(data,
                                 [my_list[0:my_list_size - 1]])  # my_list[0:my_list_size - 1] means 0 index to 59 index
                label = np.append(label,
                                  [my_list[my_list_size - 1]])  # my_list[my_list_size - 1] means 60 index(last index)

        row_num = int(len(data) / (my_list_size - 1))  # row_num = 208
        col_num = my_list_size - 1  # col_num = 60
        data = np.reshape(data, (row_num, col_num))
        data = data.astype(np.float)  # convert numpy.str_ to numpy.float64
        return data, label

    def sonarLabelToNum(self, label):
        unique_values = np.unique(label)
        class_numbers = np.arange(1, len(unique_values) + 1, 1)  # type is 'numpy.int32'
        label_num = np.ones(1 * len(label))
        for i in range(0, len(unique_values)):
            # for j in range(0,len(unique_vlaues)-1)
            indexs = np.where(label == unique_values[i])
            label_num[indexs] = class_numbers[i]
        label_num = label_num.astype(int)  # new
        return label_num

    def sonarDataNormalize(self, data):
        '''
        in this function we normalized data by ''min-max normalization bitween'' [-1,+1]
        :param data: this is a 2d nd-array and its normal bitween [0,1]
        :return:
        '''
        data_normalized = np.array([])
        [row_num, col_num] = data.shape
        max_data = data.max()  # or use this code max_data = np.amax(data)
        min_data = data.min()  # or use this code min_data = np.amin(data)
        # [a, b] = [-1, +1] rang
        a = -1
        b = +1
        for x in data:
            temp = ((b - a) * ((x - min_data) / (max_data - min_data))) + a
            data_normalized = np.append(data_normalized, [temp])
        data_normalized = np.reshape(data_normalized, (row_num, col_num))
        return data_normalized

    def sonarDataDiscretized(self, data):
        '''
        this function get data_normalized and discretized them to 5 bins (1,2,3,4,5)
        :param data:  this is a 2d nd-array and its normal bitween [-1,+1]
        :return:
        '''
        data_discretized = np.array([])
        [row_num, col_num] = data.shape
        for i in range(0, row_num):
            for j in range(0, col_num):
                x = data[i, j]
                if x <= 1 and x > 0.6:
                    data_discretized = np.append(data_discretized, [1])
                elif x <= 0.6 and x > 0.2:
                    data_discretized = np.append(data_discretized, [2])
                elif x <= 0.2 and x > -0.2:
                    data_discretized = np.append(data_discretized, [3])
                elif x <= -0.2 and x > -0.6:
                    data_discretized = np.append(data_discretized, [4])
                elif x <= -0.6 and x >= -1:
                    data_discretized = np.append(data_discretized, [5])

        data_discretized = np.reshape(data_discretized, (row_num, col_num))
        data_discretized = data_discretized.astype(int)  # new
        return data_discretized
