close all; clc; clear variables;
addpath('../');

% number of cut lines
n_cutx = 5;
n_cuty = 5;
% partition_type='debug_regular';
% partition_type='debug_random';
% partition_type='debug_random1818';
partition_type='regular';
%partition_type='random';
% partition_type='mild_random';
% partition_type='mild_random1818';
% partition_type='worst';

% create partition typpe
[X,Y,nx,ny] = create_mesh_partition(n_cutx,n_cuty,partition_type);
if ny>nx
    warning('we assume Px >= Py for angle priorities, so we need nx>=ny');
end

plot_mesh=false;
plot_dag=false;
% create adjacency matrix
[diG,order,corners]=create_adjacency_matrices(X,Y,nx,ny,plot_mesh,plot_dag);

for q=1:length(diG)
    A{q}=adjacency(diG{q});
end
% 1-direction sweep
do_plot_sweep=true;
n_angle_sets=1;


A=A{1};
n = length(A);
rela = 1e-6; abso = 1e-6; tol = abso*ones(n,1);
opts = odeset('RelTol',rela,'AbsTol',tol,'InitialStep',1e-5);
dat.max=ones(n,1);
dat.solve_speed=1;
dat.A=A;
[t,y]=ode23(@ode_sweep_fun,[0 50],zeros(n,1),opts,dat);

figure(55); hold all;
list_k=[1 2 3 4 5 6 7 35 36];
for kk=1:length(list_k)
    k=list_k(kk);
    plot(t,y(:,k));
    if kk==1
        leg=char(num2str(k));
    else
        leg=char(leg,num2str(k));
    end
end
legend(leg,'Location','Best');
grid on


