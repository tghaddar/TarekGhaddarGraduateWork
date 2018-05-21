function [flux] = CalculateAnalyticalAbsorber
%Calculates the analytical solution using quadrature.
close all; clc;

num_positive_polar=5;

psi_incident = 7/2;
x=(linspace(0,1,1000))';
scalar_flux = zeros(length(x),1);
XS=5;

for i_polar=1:num_positive_polar
    if(i_polar==1)
        leg=char(num2str(i_polar));
    else
        leg=char(leg,num2str(i_polar));
    end
    [mu,w] = lgwt(2*i_polar,-1,1);
    ind=find(mu>0);
    mu_positive=mu(ind);
    scalar_flux = zeros(length(x),1);
    for j = 1:length(mu_positive)
        mu_val = mu_positive(j);
        polar_weight = w(j);
        polar_weight
        scalar_flux = scalar_flux + psi_incident*polar_weight*exp(-XS*x/mu_val);
    end
    plot(x,scalar_flux); hold all;
end

s = psi_incident*expint(2,sym(XS*x));
exact = vpa(s,10);
plot(x,exact,'-+');
leg=char(leg,'Analytical');

legend(leg,'Location','Best')
end

