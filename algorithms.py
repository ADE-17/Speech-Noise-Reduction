import pickle
import numpy as np

class RLSFilter:
    def __init__(self, filter_order=4, delta=1.0, lambda_=0.99):
        self.filter_order = filter_order
        self.delta = delta
        self.lambda_ = lambda_
        self.weights = np.zeros(filter_order)
        self.P = np.eye(filter_order) / delta

    def train(self, noisy_signal, clean_signal):
        n = len(noisy_signal)
        output_signal = np.zeros(n)
        error_signal = np.zeros(n)

        for i in range(self.filter_order, n):
            x = noisy_signal[i-self.filter_order:i][::-1]  
            d = clean_signal[i] 
            
            y = np.dot(self.weights, x)  
            e = d - y  

            Pi = np.dot(self.P, x)
            k = Pi / (self.lambda_ + np.dot(x.T, Pi))
            
            self.P = (self.P - np.outer(k, np.dot(x.T, self.P))) / self.lambda_
            self.weights += k * e

            output_signal[i] = y
            error_signal[i] = e
        
        return output_signal, error_signal

    def train_batch(self, noisy_signals, clean_signals):
        for noisy_signal, clean_signal in zip(noisy_signals, clean_signals):
            self.train(noisy_signal, clean_signal)
    
    def apply(self, noisy_signal):
        n = len(noisy_signal)
        filtered_signal = np.zeros(n)

        for i in range(self.filter_order, n):
            x = noisy_signal[i-self.filter_order:i][::-1]  
            filtered_signal[i] = np.dot(self.weights, x)  
        
        return filtered_signal

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)


def ApplyRLS(audio, model):
    # def load_model(file_path):
    #     with open(file_path, 'rb') as file:
    #         return pickle.load(file)
    pred = model.apply(audio)
    return pred