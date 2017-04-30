//Check if all subsets meet the tolerance
while (f > tol_subset)
{
  if (f_I > tol_column)
  {
    Redistribute(X); 
  }
  if (f_J > tol_row)
  {
    Redistribute(Y);
  }
  
  Remesh;
}
