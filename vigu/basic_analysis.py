from IPython.display import display
import re

def show_columns_dtype_info(df, dtype='object'):
  '''
    type: object, number, float, int, datetime, string
  '''
  display(df.select_dtypes=[include=[dtype]].columns)


def describe_cols(df, dtype='object', transpose=False):
  if not transpose:
    df.describe(include='all')
  else:
    df.describe(include='all').transpose()