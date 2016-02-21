function [L2_error] = PlotPureScatterer( PureScatterer )
hold off;
close all; clc;
analyticalX = PureScatterer(:,1);
analyticalY = -3*analyticalX + 5;

error = abs(analyticalY-PureScatterer(:,2));
L2_error = norm(error)/norm(analyticalY);

hold on;

plot(analyticalX,analyticalY);
plot(PureScatterer(:,1),PureScatterer(:,2),'--');
legend('Analytical','PDT');
title('A 1D Pure Scatterer');
hold off;

end

