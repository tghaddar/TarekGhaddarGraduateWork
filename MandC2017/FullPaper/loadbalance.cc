//I,J subsets specified by user
//Check if all subsets meet the tolerance
while (f > tol_subset)
{
  //Mesh all subsets
  if (f_I > tol_column)
  {
    Redistribute(X); 
  }
  if (f_J > tol_row)
  {
    Redistribute(Y);
  }
}
//Remesh to get the final mesh.
