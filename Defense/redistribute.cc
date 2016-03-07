stapl::array_view num_tri_view
stapl::array_view offset_view
//offset_view stores partial sum of num_tri_view
stapl::partial_sum(num_tri_view) 
//We now have a cumulative distribution stored in offset_view
for (i = 1:X.size()-1)
{
   vector <double> pt1 = [CutLines(i-1), offset_view(i-1)]
   vector <double> pt2 = [CutLines(i), offset_view(i)]
   ideal_value = i*(N_tot/num_subsets_X);
   x_val = X-intersect(pt1,pt2,ideal_value);
   if ((x_val > x_cuts[i-1] && x_val< x_cuts[i]) 
     || equal(x_val,x_cuts[i]) || equal(x_val,x_cuts[i-1])
   {
     X[i] = x_val; 
   }
}