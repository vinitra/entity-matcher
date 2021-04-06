import pandas as pd
import sys
#
# x_file = None
#
# if len(sys.argv) > 1:
#     x_file = sys.argv[1]

x1 = pd.read_csv('X1.csv')
x2 = pd.read_csv('X2.csv')

pairs_df = pd.DataFrame(data=[], columns=['left_instance_id', 'right_instance_id'])
pairs_df.to_csv('output.csv', index=False)
