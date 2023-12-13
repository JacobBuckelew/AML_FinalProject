from torchconfig import *
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

PATH = "/home/jbuckelew/workspace/AML_FinalProject/datasets/"


def load_rds(window_size, stride_size, batch_size, num_entities):
    # Collect all the data
    path = PATH + "rds.csv"
    #label_path = PATH + "labels.csv"
    df = pd.read_csv(path)
    #labels = pd.read_csv(label_path)
    #labels = pd.DataFrame(labels["crash_utc"])
    #print(labels.index[labels['crash_utc']==1].tolist())

    T = len(df)

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime")

    # train on 60% of the time series, validate on 60-70% and test on 70-100%
    mean = df.mean(axis=0)
    std = df.std(axis=0)

    norm_df = (df - mean) / std

    train_df = norm_df.iloc[:int(0.6 * T)]
    #train_labels = labels.iloc[:int(0.6 * T)]
    val_df = norm_df.iloc[int(0.6 * T):int(0.8 * T)]
    #val_labels = labels.iloc[int(0.6 * T):int(0.8 * T)]
    test_df = norm_df.iloc[int(0.8 * T):]
    #test_labels = labels.iloc[int(0.8 * T):]
    train_labels = pd.DataFrame([0 for x in range(len(train_df))])
    val_labels = pd.DataFrame([0 for x in range(len(val_df))])
    test_labels = pd.DataFrame([0 for x in range(len(test_df))])
    train_load = DataLoader(Traffic(train_df, window_size, stride_size, num_entities, train_labels), batch_size=batch_size, shuffle=False)
    validation_load = DataLoader(Traffic(val_df, window_size, stride_size, num_entities, val_labels), batch_size=batch_size, shuffle=False)
    test_load = DataLoader(Traffic(test_df, window_size, stride_size, num_entities, test_labels), batch_size=1, shuffle=False)

    return train_load, validation_load, test_load



def load_data(window_size, stride_size, batch_size, num_entities):
    
    # Collect all the data
    path = PATH + "Inrix.csv"
    label_path = PATH + "labels.csv"
    df = pd.read_csv(path)
    labels = pd.read_csv(label_path)
    labels = pd.DataFrame(labels["crash_utc"])
    #print(labels.index[labels['crash_utc']==1].tolist())

    T = len(df)

    df["measurement_tstamp"] = pd.to_datetime(df["measurement_tstamp"])
    df = df.set_index("measurement_tstamp")

    # train on 60% of the time series, validate on 60-70% and test on 70-100%
    mean = df.mean(axis=0)
    std = df.std(axis=0)

    norm_df = (df - mean) / std

    train_df = norm_df.iloc[:int(0.6 * T)]
    train_labels = labels.iloc[:int(0.6 * T)]
    val_df = norm_df.iloc[int(0.6 * T):int(0.8 * T)]
    val_labels = labels.iloc[int(0.6 * T):int(0.8 * T)]
    test_df = norm_df.iloc[int(0.8 * T):]
    test_labels = labels.iloc[int(0.8 * T):]

    train_load = DataLoader(Traffic(train_df, window_size, stride_size, num_entities, train_labels), batch_size=batch_size, shuffle=False)
    validation_load = DataLoader(Traffic(val_df, window_size, stride_size, num_entities, val_labels), batch_size=batch_size, shuffle=False)
    test_load = DataLoader(Traffic(test_df, window_size, stride_size, num_entities, test_labels), batch_size=1, shuffle=False)

    return train_load, validation_load, test_load

class Traffic(Dataset):
    def __init__(self, data, window_size, stride_size, num_entities, labels):
        super(Traffic, self).__init__()
        self.data = data
        self.num_entities = num_entities
        self.window_size = window_size
        self.stride_size = stride_size
        self.labels = labels

        # save windows for training
        self.window_data, self.idx = self.sliding_window(data, labels)
    
    def sliding_window(self, data, labels):
        
        start = np.arange(0, len(data) - self.window_size, self.stride_size)
        return data.values, start
        
    def __len__(self):

        return len(self.idx)
    
    def __getitem__(self, idx):
        start = self.idx[idx]
        #print(start)
        #if idx == 147:
            #indices = list(self.data.index.values)
            #print(indices[start])
        end = start + self.window_size
        item = self.window_data[start:end].reshape([self.window_size, -1,1])
        if(1 in self.labels[start:end].values):
            label = 1
        else:
            label = 0
        time = self.data.iloc[start].name
        time = time.strftime('%Y-%m-%d %X')
        return torch.FloatTensor(item).transpose(0,1), label, time

