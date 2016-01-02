function [] = PlotConvergence( no_iterations, iterations,difference )

set(0,'defaultTextInterpreter','latex');

%x-axis
X_initial = [2,1.80000000000000,1.60000000000000,1.40000000000000,1.20000000000000,1,0.800000000000000,0.600000000000000,0.400000000000000,0.200000000000000,0.100000000000000,0.0800000000000000,0.0600000000000000,0.0500000000000000,0.0400000000000000,0.0300000000000000,0.0200000000000000,0.0100000000000000];

%y-axis
Y_initial = [2,3,4,5,6,7,8,9,10];

[x_plot,y_plot] = meshgrid(Y_initial,X_initial);
figure;
surface(x_plot,y_plot,no_iterations);
xlabel('$\sqrt{N}$ Total Subsets');
ylabel('Maximum Triangle Area');
title('Metric Behavior with no Load Balancing Iterations');

figure;
surface(x_plot,y_plot,iterations);
xlabel('$\sqrt{N}$ Total Subsets');
ylabel('Maximum Triangle Area');
title('Metric Behavior with 10 Load Balancing Iterations');

figure;
surface(x_plot,y_plot,difference);
xlabel('$\sqrt{N}$ Total Subsets');
ylabel('Maximum Triangle Area');
title('Metric Improvement');

end

