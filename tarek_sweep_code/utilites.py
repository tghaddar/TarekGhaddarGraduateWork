def get_ij(ss_id,numrow,numcol):
  j = ss_id % numrow
  i = (ss_id - j)/numrow
  return i,j