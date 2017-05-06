function integ = goal_function(F,x_,x,goal)

integ = abs(integral_1d_pwl_function(F,x_,x) -goal);

fprintf('Proposed x = %g, goal function = %g \n',x,integ);




