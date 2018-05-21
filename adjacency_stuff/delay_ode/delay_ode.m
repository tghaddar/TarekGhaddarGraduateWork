function [t,y]=delay_ode

close all; clc;
tic

rela = 1e-6; abso = 1e-6; tol  = abso*ones(2,1);
opts = odeset('RelTol',rela,'AbsTol',tol,'InitialStep',1e-5);
dat.max(1:2)=[1.1 2];
dat.a=1;
[t,y]=ode23(@myfun,[0 50],[0;0],opts,dat);

plot(t,y(:,1),'+-',t,y(:,2),'o-'); grid on

toc 

return
end

function dydt = myfun(time,y,dat)

dydt = zeros(length(y),1);

if y(1)<dat.max(1)
    dydt(1) = dat.a;
end
if y(2)<dat.max(2) && y(1)>=dat.max(1)
    dydt(2) = dat.a;
end

return
end
