import pandas as pd
import sys

x_files = ['X2.csv', 'X3.csv', 'X4.csv']
data = []
for file in x_files:
    # x1 = pd.read_csv('X1.csv')
    x = pd.read_csv(file)
    datapoint = [x.instance_id[0], x.instance_id[1]]
    data.append(datapoint)

print(data)
pairs_df = pd.DataFrame(data=data, columns=['left_instance_id', 'right_instance_id'])
pairs_df.to_csv('output.csv', index=False)
