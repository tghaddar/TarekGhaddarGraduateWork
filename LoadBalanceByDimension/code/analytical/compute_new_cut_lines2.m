function x_new = compute_new_cut_lines2(F,x_)

% piece-wise CDF
for i=1:length(F)
    PW_CDF(i)=sum(F(1:i));
end
PW_CDF=[0 PW_CDF]/sum(F);
% plot(x_,PW_CDF);

% we want to find the new cut lines such that each column has the same
% fraction of the total integral
n_segment   = length(x_)-1;
n_cut_lines = length(x_)-2;

x_new=x_;
for i=1:n_cut_lines
    goal = i/n_segment -eps;
    if goal<0
        error('goal is <0 (=%g) in %s',goal,mfilename);
    end
    %
    ind = find( PW_CDF < goal);
    iend=length(ind);
    if iend == length(PW_CDF)
        error('goal, %17.10e, is greater than larger value of PW_CDF, %17.7e, in %s',goal,PW_CDF(end),mfilename);
    end
    y1=PW_CDF(iend)  -goal;  x1=x_(iend);
    y2=PW_CDF(iend+1)-goal;  x2=x_(iend+1);
    slope=(y2-y1)/(x2-x1);
    b=((y1+y2)-slope*(x1+x2))/2;
    x_new(i+1)=-b/slope;
end