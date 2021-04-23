import pandas as pd

sub1 = pd.read_csv('subtask1/subtask1_t2.csv', delimiter=',')
sub2 = pd.read_csv('subtask2/subtask2_t2.csv', delimiter=',')
sub3 = pd.read_csv('subtask3/subtask3_predictions.csv', delimiter=',')

sub1.set_index("pid", inplace=True)
sub2.set_index("pid", inplace=True)
sub3.set_index("pid", inplace=True)

print(sub1)
print(sub2)
print(sub3)
print(sub1.columns)
print(sub2.columns)
print(sub3.columns)

result = pd.concat([sub1,sub2,sub3], axis = 1)
#sample = pd.read_csv('dataset/sample.csv', delimiter=',')

print(result)
print(result.columns)
filename = 'submit_7'
compression_options = dict(method='zip', archive_name=f'{filename}.csv')
result.to_csv(f'{filename}.zip', compression=compression_options, index=True, float_format='%.3f')
