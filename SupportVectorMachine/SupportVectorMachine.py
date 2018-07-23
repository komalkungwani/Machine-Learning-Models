import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


class SVM:
    def __init__(self,  visualization=True):
        self.visualization = visualization
    
    def fit(self, features, labels):
        transforms = [[1,1], [-1,1], [-1,-1], [1,-1]]
        
        all_data = features.flatten()
        opt_dict = {}

        uniq = np.unique(labels)
        uniq_features = []
        for lab in uniq:
            f = []
            for i in range(len(labels)):
                if(labels[i] == lab):
                    f.append(features[i])
            uniq_features.append(f)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      #point of expense:
                      self.max_feature_value * 0.001]

        b_range_multiple = 5

        b_multiple = 5

        latest_optimum = self.max_feature_value*10
        a=[]
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),self.max_feature_value*b_range_multiple,step*b_multiple):
                     for transformation in transforms:
                         w_t = w*transformation
                         found_option = True

                         for i in range(len(uniq)):
                             for xi in uniq_features[i]:
                                 yi = uniq[i]
                                 cal = yi*(np.dot(w_t,xi)+b)
                                 if(not cal >= 1):
                                     found_option = False
                                     break
                                 a.append((xi, ':', cal))

                         
                     if(found_option):
                         opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if(w[0] <0):
                        optimized = True
                        print("optimized a step")
                else:
                        w = w-step

                norms =sorted([n for n in opt_dict])
                opt_choice = opt_dict[norms[0]]

                self.w = opt_choice[0]
                self.b = opt_choice[1]
                latest_optimum = opt_choice[0][0]+step*2
            
        

    def predict(self, features):
        prediction = np.sign(np.dot(np.array(features),self.w)+self.b)
        return prediction.astype(int)
