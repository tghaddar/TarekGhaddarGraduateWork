function [L2_error] = PlotPureAbsorber( PureAbsorber1,PureAbsorber2,PureAbsorber3,PureAbsorber4)
hold off;
close all; clc;
analyticalX = PureAbsorber1(:,1);
s = 3.5*expint(2,sym(5*analyticalX));
flux = vpa(s,10);

%flux = CalculateAnalyticalAbsorber(1,20,analyticalX);

error = abs(flux-PureAbsorber1(:,2));
L2_error = vpa(norm(error)/norm(flux),10);

hold on;

plot(analyticalX,flux);
plot(PureAbsorber1(:,1),PureAbsorber1(:,2),'--');
% plot(PureAbsorber2(:,1),PureAbsorber2(:,2),'--');
% plot(PureAbsorber3(:,1),PureAbsorber3(:,2),'--');
% plot(PureAbsorber4(:,1),PureAbsorber4(:,2),'--');

%legend('Analytical','PDT_{pglc70}','PDT_{pglc10}','PDT_{pglc5}','PDT_{pglc1}');
legend('Analytical','PDT_{pglc70}');
title('A Pure Absorber 1D Slab with Different Angular Refinements');
xlabel('Distance (cm)');
ylabel('Flux (neutrons/cm^{2}-s'); 
hold off;

end

