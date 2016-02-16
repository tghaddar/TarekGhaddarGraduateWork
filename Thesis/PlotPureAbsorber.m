function [L2_error,flux] = PlotPureAbsorber( PureAbsorber1)
hold off;
analyticalX = PureAbsorber1(:,1);
% s = 3.5*expint(2,sym(5*analyticalX));
% analyticalY = vpa(s,10);

flux = CalculateAnalyticalAbsorber(2,1,analyticalX);

error = abs(flux-PureAbsorber1(:,2));
L2_error = norm(error)/norm(flux);

hold on;

plot(analyticalX,flux);
plot(PureAbsorber1(:,1),PureAbsorber1(:,2),'--');
% plot(PureAbsorber2(:,1),PureAbsorber2(:,2),'--');
% plot(PureAbsorber3(:,1),PureAbsorber3(:,2),'--');
% plot(PureAbsorber4(:,1),PureAbsorber4(:,2),'--');
% plot(PureAbsorber5(:,1),PureAbsorber5(:,2),'--');

legend('Analytical','PDT_{pglc}');
title('A Pure Absorber 1D Slab');
hold off;

end

