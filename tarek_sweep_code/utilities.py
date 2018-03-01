import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def get_ij(ss_id,numrow,numcol):
  j = int(ss_id % numrow)
  i = int((ss_id - j)/numrow)
  return i,j