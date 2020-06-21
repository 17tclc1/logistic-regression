import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sigmoid = Logistic Function = trả về xác suất ( từ 0 đến 1 )
# z = W0+W1.X+W2.Y
def sigmoid(data,weights):
    z = np.dot(data,weights)
    return 1.0/(1 + np.exp(-z))

def cost_function(data,RealOutput,weights):
    n = len(RealOutput)
    PredictedOutput = sigmoid(data,weights)

    #! Cost function 
    cost_class1 = -(RealOutput)*(np.log(PredictedOutput))
    cost_class0 = -(1 - RealOutput)*(np.log(1-PredictedOutput))

    cost = np.sum((cost_class1 + cost_class0)) / n

    return cost

def train(data, RealOutput, weights, learning_rate, iteration):
    cost_history = []
    for i in range(iteration):
        # Update weight
        n = len(RealOutput)
        PredictedOutput = sigmoid(data,weights)
        #! Gradient Descent 
        weights = weights - (np.dot(data.T,(PredictedOutput - RealOutput)) * learning_rate ) / n
        # Get Cost function
        cost = cost_function(data, RealOutput, weights)
        cost_history.append(cost)

        if cost < 0.25:
            return weights, cost_history

    return weights, cost_history
if __name__ == "__main__":
    dataFromFile = pd.read_csv('data.csv', header = None)

    true_x = [] # Trục x của những giá trị là 1
    true_y = [] # Trục y của những giá trị là 1
    false_x = [] # Trục x của những giá trị là 0
    false_y = [] # Trục y của những giá trị là 0
    maxvalue1 = 0
    maxvalue2 = 0

    for item in dataFromFile.values:
        if item[2] == 1.:
            true_x.append(item[0])
            true_y.append(item[1])
            if item[0] > maxvalue1:
                maxvalue1 = item[0]
        else:
            false_x.append(item[0])
            false_y.append(item[1])
            if item[1] > maxvalue2:
                maxvalue2 = item[1]
    # RealOutput là giá trị rớt môn hay không ( 0 hoặc 1 ) đọc từ file CSV
    RealOutput = np.zeros((100,1))
    data = np.zeros((100,3))
    # Khởi tạo weight
    weight = [[1.0],[1.0],[1.0]]
    i = 0
    
    for item in dataFromFile.values:
        data[i][0] = 1
        data[i][1] = item[0]
        data[i][2] = item[1]
        RealOutput[i][0] = item[2]
        i = i + 1
    weight,cost_history = train(data,RealOutput,weight,0.001,350000)   

    plt.scatter(true_x,true_y, marker = 'o', c = 'g',label = 'Passed')
    plt.scatter(false_x, false_y, marker = 's', c = 'r',label = 'Failed')

    plt.xlabel("Test1")
    plt.ylabel("Test2")

    # Test dữ liệu mới
    if(sigmoid([1,80,80],weight) > 0.5):
        print("1")
    else:
        print("0")

    x2 = (- weight[0] - weight[1]*maxvalue1) / weight[2]
    x2_1 = (-weight[0] - weight[1]*maxvalue2)/ weight[2]

    plt.plot([maxvalue1,x2],[x2_1,maxvalue2],label = 'Seperation')

    plt.legend()
    plt.show()
