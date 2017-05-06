function integ = integral_1d_pwl_function(F,x,upper)

ind=find(x<=upper-eps);
iend=length(ind);

if iend == length(x)
    error('upper bound, %17.10e, is greater than larger value of x, %17.7e, in %s',upper,x(end),mfilename);
end

integ=0;
for i=1:iend
    F1=F(i);   x1=x(i);
    F2=F(i+1); x2=x(i+1);
    if i==iend
        x2=upper;
    end
    fh = @(x) F2*(x-x1)/(x2-x1) + F1*(x-x2)/(x1-x2); 
    integ = integ + integral(fh,x1,x2);
end

fprintf('Proposed x = %g, integral = %g \n',upper,integ);
