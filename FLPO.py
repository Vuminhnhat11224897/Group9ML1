from sklearn.ensemble import RandomForestClassifier
from collections import deque
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.io import arff
import matplotlib.pyplot as plt

# Load data
data = arff.loadarff('sea.arff')
df = pd.DataFrame(data[0])
df['cl'] = df['cl'].str.decode('utf-8').map({'0': 0, '1': 1})

# Train Random Forest Classifier
def train_random_forest_classifier(X, y, n_estimators=50):
    """
    Huấn luyện bộ phân loại Random forest trên dữ liệu X, y.
    n_estimators: Số lượng cây trong Random forest.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(X, y)
    return rf

# Create data stream
def create_datastream(data, batch_size):
    datastream = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        datastream.append(batch)
    return datastream

# Adaptive ensemble size
def adaptive_ensemble_size(C, sample, unique_label, anpha, min_num=3):
    probability_list = []
    for i in range(len(unique_label)):
        li = [] 
        for j in range(len(C)):
            probabilities = C[0].predict_proba(sample)[0]
            probability_dict = {label: prob for label, prob in zip(C[0].classes_, probabilities)}
            latest_proba = probability_dict.get(unique_label[i], 0)
            probabilities1 = C[j].predict_proba(sample)[0]
            probability_dict1 = {label: prob for label, prob in zip(C[j].classes_, probabilities1)}
            current_proba = probability_dict1.get(unique_label[i], 0)
            if len(li) < min_num:
                li.append(current_proba)
            else:
                if abs(current_proba - latest_proba) < anpha:
                    li.append(current_proba)
                else:
                    break
        probability_list.append(li)
    return probability_list

# Linear regression
def linear_regression(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

# Tendency prediction
def tendency_prediction(probability_list, Y, epsilon=0.01):
    predicted_probabilities = []
    for i in range(len(Y)):
        li = probability_list[i]
        x = np.arange(1, len(li) + 1)
        y = np.array(li)
        slope, intercept = linear_regression(x, y)
        next_value = slope * (len(li) + 2) + intercept
        li.insert(0, next_value)
        weights = np.array([1 / (x + 1) for x in range(len(li))])
        weighted_prob = sum([li[x] * weights[x] for x in range(len(li))]) / sum(weights)
        predicted_probabilities.append(weighted_prob)
    Ps = Y[np.argmax(predicted_probabilities)]
    return Ps

# Process data stream
def process_data_stream(S, m, k, unique_labels):
    C = deque(maxlen=m)
    true_labels = []
    predicted_labels = []
    pre = []
    block_accuracies = []  # Danh sách để lưu trữ độ chính xác của từng block

    for i in range(len(S) - 1):
        Bi = S[i]
        block_predictions = []
        X = Bi.iloc[:, :-1]
        y = Bi.iloc[:, -1]
        Ci = train_random_forest_classifier(X, y)
        C.append(Ci)
        if len(C) < k:
            continue
        Bi_1 = S[i + 1]
        for index, row in Bi_1.iterrows():
            sample = pd.DataFrame([row[:-1]], columns=Bi.columns[:-1])
            anpha = (1500 / len(Bi_1)) * 0.2
            selected_classifiers = adaptive_ensemble_size(C, sample, unique_labels, anpha)
            pre_sample = tendency_prediction(selected_classifiers, unique_labels)
            block_predictions.append(pre_sample)
            true_labels.append(row.iloc[-1])
            predicted_labels.append(pre_sample)
        pre.append(block_predictions)

        # Tính toán độ chính xác cho block hiện tại
        block_accuracy = accuracy_score(true_labels[-len(Bi_1):], predicted_labels[-len(Bi_1):])
        block_accuracies.append(block_accuracy)
    
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Accuracy: {accuracy}")

    # Trực quan hóa độ chính xác của từng block
    plt.figure(figsize=(10, 6))
    plt.plot(block_accuracies, marker='o', linestyle='-', color='b')
    plt.title('Accuracy of Each Block')
    plt.xlabel('Block Index')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# Main execution
batch_size = 1500
S = create_datastream(df, batch_size)
m = 15
k = 3
unique_labels = list(set(df.iloc[:, -1]))
process_data_stream(S, m, k, unique_labels)