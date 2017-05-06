function x_new = compute_new_cut_lines(F,x_)

% for i=1:length(F)
%     PWL_F(i)=sum(F(1:i));
% end
% PWL_F=[0 PWL_F];
    
% compute integral of the piece-wise function
integ = integral_1d_pwl_function(F,x_,x_(end));

% we want to find the new cut lines such that each column has the same
% fraction of the total integral
n_segment = length(x_)-1;
x_new=x_;
for i=1:n_segment-1
    goal = i*integ/n_segment;
    % guess 
    upper = x_(i+1);
    fprintf('i=%d, goal=%g, upper=%g,x_new(i)=%g \n',i,goal,upper,x_new(i));
    gh = @(x) abs(integral_1d_pwl_function(F,x_,x)-goal);
    % gh = @(x) goal_function(F,x_,x,goal_);
    options = optimset('TolFun',1e-7,'TolX',1e-7);
    [out,fval,exitflag,output] = fminsearchbnd(gh,upper,x_new(i),x_(end),options);   
    x_new(i+1)=out;
end