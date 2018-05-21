function dydt = ode_sweep_fun(t,y,dat)

n = length(y);
dydt = zeros(n,1);
A = dat.A;
c = dat.solve_speed;

if y(1)<dat.max(1)
    dydt(1) = c;
end

for k=2:n   
    % find nonzero entries in column above diagonal
    ind = find(A(1:k-1,k)>0);
    % if predecessor reached their max values, then activate
    if isempty(ind) 
        if y(k)<dat.max(k)
            dydt(k) = c;
        end
    else
        if all(y(ind)>=dat.max(ind)) && y(k)<dat.max(k)
            dydt(k) = c;
        end
    end
end

return
end