//This is pseudocode for the load balancing by dimension algorithm.
function LOAD_BALANCE_BY_DIMENSION()
{
  //All subsets must be meshed initially. 
  MESH all subsets;
  
  //Caluclate the column-wise load balance metric, f_I
  CALCULATE f_I;
  //Calculate the row-wise load balance metric
  CALCULATE f_J;
  
  if (f_I <= f_J)
  {
    while (f_I > column_tolerance)
    {
      //If the column wise metric is still greater than the tolerance, 
      Redistribute(x_cuts);
      //Remesh 
      MESH all subsets;
    }
  }
  else
  {
    while(f_J > row_tolerance)
    {
      //If the row-wise metric is still greater than the tolerance.
      Redistribute(y_cuts);
      //Remesh
      MESH all subsets;
    }
  }
  
  //Once we have columns (rows) balanced, we have to load balance the rows(columns) in each column(row) now.
  for each column(row)
  {
    //Calculate the row-wise(column-wise) metric for each column(row)
    CALCULATE f_I(f_J);
    while (f_I(f_J) > tolerance)
    {
      //Redistribute the rows in this column.
      Redistribute(y_cuts_column(x_cuts_row));
      //All subsets must be meshed initially. 
      MESH all subsets;
    }
  }
}

//Psuedocode to build and store and undirected task dependence graph.
function BUILD_UNDIRECTED_TDG(x_cuts,y_cuts)
{
  //We change how the y_cuts(x_cuts) are stored. They are now stored BY COLUMN(ROW). We need to store them this way in order to properly build our mesh, and properly define subset boundaries.
  //Each subset knows it's own xmin/ymin, xmax/ymax boundaries. We can base this on a loose i,j framework.
  //We know that each subset will have a neighbor above and below it (except if it's on a global y boundary). These neighbors never change and we do not need to account for them.
  //The only neighbors in flux after a load balancing iteration are the subsets right and/or left neighbors.

  //We should be able to replace column with row in the case that the rows have cuts going across the entirety of the domain.
  
  //If the current subset is in the first column, we only need to look to our right neighbor column.
  if (column == first_column)
  {
    //Pull the y_cuts for the second column.
    second_column_y_cuts = y_cuts[second_column];
    //Loop over all the y_cuts in this column.
    for (i = 0; i < num_rows-1; ++i)
    {
      //Check if this y_cut is less than or equal to the maximum y boundary of this subset.
      if ( second_column_y_cuts[i] < y_max )
      {
        //Check if this y_cut is greater than or equal to the minimum y boundary of this subset.
        if ( second_column_y_cuts[i] > y_min )
	{
	  //Register both subsets that share that y_cut as neighbors.
	}
	else if ( second_column_y_cuts[i] == y_min )
	{
	  //Register the subset that has this y_cut as its ymin as a neighbor.
	}
      }
      else if ( second_column_y_cuts[i] == y_max)
      {
        //Register the subset that has this y_cut as its ymax as a neighbor. 
      } 
      //This y_cut is greater than the maximum y boundary of this subset.
      else
      {
	//Check if we're looking at the first cut line in the column.
	if ( i == 0 )
	{
          //Register the subset that has that y_cut as its ymax as a neighbor.
	}
	else if(second_column_y_cuts[i-1] < y_max)
	{
	  //Register the subset that has second_colum_y_cut[i] as its ymax as a neighbor.
	}
      }
    }
  }
  //If the subset is in the last column, we only need to look to our left neighbor column.
  else if(column == last_column)
  { 
    //Pull the y_cuts for the second to last column.
    left_column_y_cuts = y_cuts[last_column-1];
    //Loop over all the y_cuts in this column.
    for (i = 0; i < num_rows-1; ++i)
    {
      //Check if this y_cut is less than or equal to the maximum y boundary of this subset.
      if ( left_column_y_cuts[i] < y_max )
      {
        //Check if this y_cut is greater than or equal to the minimum y boundary of this subset.
        if ( left_column_y_cuts[i] > y_min )
	{
	  //Register both subsets that share that y_cut as neighbors.
	}
	else if ( left_column_y_cuts[i] == y_min )
	{
	  //Register the subset that has this y_cut as its ymin as a neighbor.
	}
      }
      else if ( left_column_y_cuts[i] == y_max)
      {
        //Register the subset that has this y_cut as its ymax as a neighbor. 
      } 
      //This y_cut is greater than or equal to the maximum y boundary of this subset.
      else
      {
	//Check if we're looking at the first cut line in the column.
	if ( i == 0 )
	{
          //Register the subset that has that y_cut as its ymax as a neighbor.
	}
	else if(left_column_y_cuts[i-1] < y_max)
	{
	  //Register the subset that has second_colum_y_cut[i] as its ymax as a neighbor.
	}
      }
    }
  }
  //If the current subset is located in an interior column, we need to check both the right and left neighbor columns.
  else
  {
    left_column_y_cuts = y_cuts[current_column - 1];
    right_column_y_cuts = y_cuts[current_column + 1];
    //Loop over all the y_cuts in this column.
    for (i = 0; i < num_rows-1; ++i)
    {
      //Check if this y_cut is less than or equal to the maximum y boundary of this subset.
      if ( left_column_y_cuts[i] < y_max )
      {
        //Check if this y_cut is greater than or equal to the minimum y boundary of this subset.
        if ( left_column_y_cuts[i] > y_min )
	{
	  //Register both subsets that share that y_cut as neighbors.
	}
	else if ( left_column_y_cuts[i] == y_min )
	{
	  //Register the subset that has this y_cut as its ymin as a neighbor.
	}
      }
      else if ( left_column_y_cuts[i] == y_max)
      {
        //Register the subset that has this y_cut as its ymax as a neighbor. 
      } 
      //This y_cut is greater than the maximum y boundary of this subset.
      else
      {
	//Check if we're looking at the first cut line in the column.
	if ( i == 0 )
	{
          //Register the subset that has that y_cut as its ymax as a neighbor.
	}
	else if(left_column_y_cuts[i-1] < y_max)
	{
	  //Register the subset that has second_colum_y_cut[i] as its ymax as a neighbor.
	}
      }
      //Check if this y_cut is less than or equal to the maximum y boundary of this subset.
      if ( right_column_y_cuts[i] < y_max )
      {
        //Check if this y_cut is greater than or equal to the minimum y boundary of this subset.
        if ( right_column_y_cuts[i] > y_min )
	{
	  //Register both subsets that share that y_cut as neighbors.
	}
	else if ( right_column_y_cuts[i] == y_min )
	{
	  //Register the subset that has this y_cut as its ymin as a neighbor.
	}
      }
      else if ( right_column_y_cuts[i] == y_max)
      {
        //Register the subset that has this y_cut as its ymax as a neighbor. 
      } 
      //This y_cut is greater the maximum y boundary of this subset.
      else
      {
	//Check if we're looking at the first cut line in the column.
	if ( i == 0 )
	{
          //Register the subset that has that y_cut as its ymax as a neighbor.
	}
	else if( right_column_y_cuts[i-1] < y_max )
	{
	  //Register the subset that has second_colum_y_cut[i] as its ymax as a neighbor.
	}
      }
    }
  }
  //Each subset cleans up it's duplicates and we now have a map of the neighbors for each subset!
}
