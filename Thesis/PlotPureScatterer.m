function [L2_error] = PlotPureScatterer( PureScatterer )
hold off;
close all; clc;
analyticalX = PureScatterer(:,1);

XS = 100;
D = 1/(3*XS);
incident = 7;
A = -incident/(1+4*D);
B = incident + 2*D*A;


analyticalY = A*analyticalX + B;

error = abs(analyticalY-PureScatterer(:,2));
L2_error = norm(error)/norm(analyticalY);

hold on;

plot(analyticalX,analyticalY);
plot(PureScatterer(:,1),PureScatterer(:,2),'--');
legend('Analytical','PDT');
title('A 1D Pure Scatterer');
xlabel('Distance (cm)');
ylabel('Flux (neutrons/cm^{2}-s)');

hold off;

end


