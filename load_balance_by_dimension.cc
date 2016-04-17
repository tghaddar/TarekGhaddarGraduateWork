//This is pseudocode for the load balancing by dimension algorithm.

MESH all subsets;

//Caluclate the column-wise load balance metric, f_I
CALCULATE f_I;

while (f_I > column_tolerance)
{
  //If the column wise metric is still greater than the tolerance, 
  Redistribute(X);
}

//Once we have columns balanced, we have to load balance the rows in each column now.

for each column
{
  //Calculate the general, per subset  metric for the column.
  CALCULATE f;
  while (f > tolerance)
  {
    //Redistribute the rows in this column.
    Redistribute(Y);
  }
}




