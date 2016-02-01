function [L2_error] = PlotPureAbsorber( PureAbsorber )
hold off;
analyticalX = PureAbsorber(:,1);
s = 3.5*expint(2,sym(5*analyticalX));
analyticalY = vpa(s,10);

error = abs(analyticalY-PureAbsorber(:,2));
L2_error = norm(error);

hold on;

plot(analyticalX,analyticalY);
plot(PureAbsorber(:,1),PureAbsorber(:,2),'--');
legend('Analytical','PDT');
hold off;

end

