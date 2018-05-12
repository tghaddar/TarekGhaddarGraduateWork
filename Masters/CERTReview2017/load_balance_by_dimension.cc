while (f_I > tol_column)
{
  if (f_I > tol_column)
  {
    Redistribute(X); 
  }
  
  Remesh;
}

while (f > tol)
{
  for i < I
  {
    if (f_J[i] > tol_row)
    {
      Redistribute(Y_i);
    }
  }
  Remesh;
}
