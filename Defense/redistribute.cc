//stores number of triangles for each row/col 
num_tri_view
//stores the partial sum of num_tri_view
offset_view
//We now have a cumulative distribution stored in offset_view
for (i = 1:X.size()-1)
{
   pt1 = [X(i-1), offset_view(i-1)]
   pt2 = [x(i), offset_view(i)]
   ideal_num_triangles = i*(N_tot/num_subsets_X);
   x_val = X-intersect(pt1,pt2,ideal_value);
   //The cut line in question has been redistributed.  
   X[i] = x_val; 
}
