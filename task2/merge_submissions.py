import pandas as pd

sub1 = pd.read_csv('subtask1/subtask1_t2.csv', delimiter=',')
sub2 = pd.read_csv('subtask2/subtask2_t2.csv', delimiter=',')
sub3 = pd.read_csv('subtask3/subtask3_predictions.csv', delimiter=',')

print(sub1)
print(sub2)
print(sub3)
print(sub1.columns)
print(sub2.columns)
print(sub3.columns)

#sample = pd.read_csv('dataset/sample.csv', delimiter=',')

result_tmp = pd.concat([sub1, sub2.iloc[:,1:], sub3.iloc[:,1:]], axis=1)
#print(result_tmp.columns)
#result_tmp=result_tmp.set_index('pid')
print(result_tmp)
print(result_tmp.columns)
#result_tmp = result_tmp.reindex(sample['pid'].to_numpy())
#result_tmp.reindex(sample['pid'].to_numpy())
#print(result_tmp)
#print(result_tmp)
#print(result_tmp)

#result_tmp = result_tmp.set_index(result_tmp['pid'].to_numpy(),drop = False)
#print(result_tmp)
#print(result_tmp.columns)
    
filename = 'submit_6'
compression_options = dict(method='zip', archive_name=f'{filename}.csv')
result_tmp.to_csv(f'{filename}.zip', compression=compression_options, index=False, float_format='%.3f')
