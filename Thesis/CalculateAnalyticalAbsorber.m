function [flux] = CalculateAnalyticalAbsorber( num_angles_polar,num_angles_az, analyticalX )
%Calculates the analytical solution using quadrature.

[nodes,weights] = lgwt(2*num_angles_polar,-1,1);
psi_incident = 7;

weight_az = 2*pi/(4*num_angles_az);
weight_az

flux = zeros(length(analyticalX),1);

for i = 1:length(analyticalX)
   x_val = analyticalX(i);
   flux_val = 0;
   for j = 1:length(nodes)
      mu_val = abs(nodes(j));
      polar_weight = weights(j);
      flux_val = flux_val +polar_weight*weight_az*exp(-5*x_val/mu_val);
   end
   
   flux(i) = psi_incident * flux_val;
    %flux(i) = flux_val;
end



end

