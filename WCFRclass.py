from pyitlib import discrete_random_variable as drv
import numpy as np
import math


class WCFRclass:
    dataset = np.array([])
    data_discretized = np.array([])
    label_num = np.array([])
    MI = np.array([])
    n = 0  # number of features
    S = np.array([])
    S_vector_size = 0  # candidate features
    k = 0  # number of selected features
    New_Data_Discretized = np.array([])

    def __init__(self, data, data_discretized, label_num):
        self.setDatasetForSonarData(data)
        self.setdata_discretizedForSonarData(data_discretized)
        self.setLabel_numForSonarData(label_num)
        self.mutualInformationForSonarData()
        self.setnForSonarData(data)
        self.setSsizeForSonarData(data)

    def setDatasetForSonarData(self, data):
        self.dataset = data

    def setdata_discretizedForSonarData(self, data_discretized):
        self.data_discretized = data_discretized

    def setLabel_numForSonarData(self, label_num):
        self.label_num = label_num

    def setnForSonarData(self, data):
        number_of_features = np.size(data, 1)
        self.n = number_of_features

    def setSsizeForSonarData(self, data):
        feature_vector_size = np.size(data, 1)
        if feature_vector_size >= 50:
            self.S_vector_size = 50
            self.k = 50
        else:
            self.S_vector_size = feature_vector_size
            self.k = feature_vector_size

    def setSmanual(self, vector):
        self.S = vector
        '''
        for sonar data first big MI is index 11 in MI. MI index start from 0 to 59 that means best feature for S is
        feature number 12. 
        '''

    def setNewDataDiscretized(self):
        temp = np.zeros((np.size(self.data_discretized, 0), self.S_vector_size))
        count = 0
        for i in self.S:
            temp[:, count] = self.data_discretized[:, i]
            count = count + 1

        self.New_Data_Discretized = temp
        self.New_Data_Discretized = self.New_Data_Discretized.astype(np.int64)

        # this part of code use for check
        '''
        sampel_num = 0
        print(np.shape(self.New_Data_Discretized))
        for i in range(0, np.size(self.data_discretized, 0)):
            count = 0
            for j in range(0, 50):
                if (self.data_discretized[i, self.S[j]] == self.New_Data_Discretized[i, j]):
                    count = count + 1
                    print(self.data_discretized[i, self.S[j]])
                    print(self.New_Data_Discretized[i, j])
                    print("=======")
                if count == 50:
                    print("sampel pass")
                    sampel_num = sampel_num + 1
        print(sampel_num)
        '''

    def mutualInformationForSonarData(self):  # data_discretized, label_num
        y = self.label_num
        x = self.data_discretized
        MI1 = np.array([])
        for i in range(0, np.size(x, 1)):
            temp = drv.information_mutual(x[:, i], y, cartesian_product=False)
            MI1 = np.append(MI1, [temp])
        MI2 = np.array([])
        temp_x = x.T
        MI2 = np.append(MI2, [drv.information_mutual(temp_x, y, cartesian_product=True)])
        self.MI = MI1
        return MI1  # you can use MI2 except of MI1

    def WCRFalgoritmForSonarData(self):  # , data_discretized, label_num
        S = np.array([])
        S = S.astype(int)
        F = np.array([i for i in range(1, 1 + np.size(self.data_discretized, 1))])  # np.size(data_discretized, 1) = 60
        # F index start from 0 to 59
        # print(F[0]) # == 1
        # print(F[59]) # == 60
        # print(np.size(F)) # == 60
        MI = self.mutualInformationForSonarData()  # data_discretized, label_num
        # np.size(MI, 1) = 60
        # MI index start from 0 to 59
        f_new = np.argmax(MI)
        S = np.append(S, [f_new])
        self.S = S
        F[f_new] = -1
        count = 1
        k = self.k
        n = self.n
        # print(n)
        print(self.S)
        while count < k:
            temp = -1
            best_index = -1
            l = n - count  # notic that n = number of featurs = (l + count)
            for m in range(0, n):  # range(1, l+1)
                # print("m is :" + str(m))
                if F[m] != -1:
                    for i in range(0, count):
                        # print("i is :" + str(i))
                        x_m_index = m
                        h = self.Jwcfr(x_m_index)
                        print("jwcfr")
                        print(h)
                        if (h > temp):
                            temp = h
                            best_index = x_m_index
            f_new = best_index
            print("f_new")
            print(f_new)
            S = np.append(S, [f_new])
            self.S = S
            F[f_new] = -1
            print("S")
            print(self.S)
            print("F")
            print(F)
            print("==============")
            print("==============")
            print("==============")
            count = count + 1

    def Jwcfr(self, x_m_index):
        sigma1 = 0.0
        sigma2 = 0.0
        u1 = 0.0
        u2 = 0.0

        # compute u1
        for i in range(0, np.size(self.S)):
            x_i_index = self.S[i]
            temp = drv.information_mutual_conditional(self.data_discretized[:, x_m_index], self.label_num,
                                                      self.data_discretized[:, x_i_index], cartesian_product=False)
            u1 = u1 + temp
        u1 = u1 / np.size(self.S)

        # compute sigma1
        for i in range(0, np.size(self.S)):
            x_i_index = self.S[i]
            temp = drv.information_mutual_conditional(self.data_discretized[:, x_m_index], self.label_num,
                                                      self.data_discretized[:, x_i_index], cartesian_product=False)
            sigma1 = sigma1 + ((temp - u1) ** 2)
        sigma1 = sigma1 / np.size(self.S)
        sigma1 = math.sqrt(sigma1)

        # compute u2
        for i in range(0, np.size(self.S)):
            x_i_index = self.S[i]
            temp1 = drv.information_mutual(self.data_discretized[:, x_m_index], self.data_discretized[:, x_i_index],
                                           cartesian_product=False)
            temp2 = drv.information_mutual_conditional(self.data_discretized[:, x_m_index],
                                                       self.data_discretized[:, x_i_index],
                                                       self.label_num, cartesian_product=False)
            u2 = u2 + (temp1 - temp2)
        u2 = u2 / np.size(self.S)

        # compute sigma1
        for i in range(0, np.size(self.S)):
            x_i_index = self.S[i]
            temp1 = drv.information_mutual(self.data_discretized[:, x_m_index], self.data_discretized[:, x_i_index],
                                           cartesian_product=False)

            temp2 = drv.information_mutual_conditional(self.data_discretized[:, x_m_index],
                                                       self.data_discretized[:, x_i_index],
                                                       self.label_num, cartesian_product=False)

            sigma2 = sigma2 + ((temp1 - temp2 - u2) ** 2)

        sigma2 = sigma2 / np.size(self.S)
        sigma2 = math.sqrt(sigma2)

        temp = 0
        temp1 = 0
        temp2 = 0
        # compute summation of (I(x_m;C|x_s)) for each (x_s) in S
        eq_part1 = 0
        for i in range(0, np.size(self.S)):
            x_i_index = self.S[i]
            temp = drv.information_mutual_conditional(self.data_discretized[:, x_m_index], self.label_num,
                                                      self.data_discretized[:, x_i_index], cartesian_product=False)
            eq_part1 = eq_part1 + temp

        # compute summation of (I(x_m;x_s) - I(x_m;x_s|C)) for each (x_s) in S
        eq_part2 = 0
        for i in range(0, np.size(self.S)):
            x_i_index = self.S[i]
            temp1 = drv.information_mutual(self.data_discretized[:, x_m_index], self.data_discretized[:, x_i_index],
                                           cartesian_product=False)
            temp2 = drv.information_mutual_conditional(self.data_discretized[:, x_m_index],
                                                       self.data_discretized[:, x_i_index],
                                                       self.label_num, cartesian_product=False)
            eq_part2 = eq_part2 + (temp1 - temp2)

        J_wcrf = ((1 - sigma1) * (eq_part1)) - ((1 - sigma2) * (eq_part2))

        return J_wcrf

    # def conditionalMutualInformationForSonarData(self, x_m_index, x_i_index):
    #
    #     # MI1 = np.array([])
    #     temp = drv.information_mutual_conditional(self.data_discretized[:, x_m_index], self.label_num,
    #                                               self.data_discretized[:, x_i_index],
    #                                               cartesian_product=False)
    #     # MI1 = np.append(MI1, [temp])
    #     return temp
