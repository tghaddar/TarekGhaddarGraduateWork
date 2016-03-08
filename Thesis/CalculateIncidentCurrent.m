%Calculate the incident current.



num_angles = [1 2 5 10 20 40 50 60 70]';
all_j_inc = zeros(length(num_angles),2);
all_j_inc(:,1) = num_angles;


for i = 1:length(num_angles);
    num_positive_polar = num_angles(i);
    j_inc = 0;
    psi_incident = 7/2;

    [mu,w] = lgwt(2*num_positive_polar,-1,1);
    ind=find(mu>0);
    mu_positive=mu(ind);

    for j = 1:length(mu_positive)
        mu_val = mu_positive(j);
        polar_weight = w(j);
        j_inc = j_inc + psi_incident*polar_weight*mu_val;
    end
   all_j_inc(i,2) = j_inc; 
end
