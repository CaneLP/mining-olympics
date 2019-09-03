from data_processing import *


pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)

data = get_data(printable=False, merge_noc=True)
data = fill_na_attr_rows(data)
data.to_csv(r'../data/full.csv')



