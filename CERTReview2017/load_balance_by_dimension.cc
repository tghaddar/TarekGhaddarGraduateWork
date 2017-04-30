while (f_I > tol_column)
{
  if (f_I > tol_column)
  {
    Redistribute(X); 
  }
  
  Remesh;
}

while (f > tol)
  if (f_J > tol_row)
  {
    Redistribute(Y);
  }
  Remesh;
}
