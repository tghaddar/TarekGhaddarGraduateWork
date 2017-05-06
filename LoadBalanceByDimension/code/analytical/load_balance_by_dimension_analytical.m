clear variables; close all; clc;

% define function
% function_type='gaussian';
% function_type='constant';
% function_type='linearx';
% function_type='lineary';
% function_type='opposite_gaussian';
function_type='double_gaussian';
switch lower(function_type)
    case 'gaussian'
        % Gaussian function
        Lx=2;
        Ly=2;
        x0=.25*Lx;
        y0=.25*Ly;
        sigma=(Lx*Ly)/500;
        fh = @(x,y) (x/Lx).*(1-(x/Lx)).*(y/Ly).*(1-(y/Ly)).*exp(-((x-x0).^2+(y-y0).^2)/sigma);
    case 'constant'
        % constant mesh density
        fh = @(x,y) 1+0*x+0*y;
        Lx=1;Ly=2;
    case 'linearx'
        % linear in x mesh density
        fh = @(x,y) 1*x+0*y;
        Lx=1;Ly=2;
    case 'lineary'
        % linear in y mesh density
        fh = @(x,y) 0*x+1*y;
        Lx=1;Ly=2;
    case 'opposite_gaussian'
        % Gaussian function
        Lx=2;
        Ly=2;
        x0=.25*Lx;
        y0=.25*Ly;
        sigma0=(Lx*Ly)/500;
        x1=.75*Lx;
        y1=.75*Ly;
        sigma1=(Lx*Ly)/500;
        fh = @(x,y) (x/Lx).*(1-(x/Lx)).*(y/Ly).*(1-(y/Ly)).*(exp(-((x-x0).^2+(y-y0).^2)/sigma0)...
            +exp(-((x-x1).^2+(y-y1).^2)/sigma1));
    case 'double_gaussian'
        % Gaussian function
        Lx=2;
        Ly=2;
        x0=.25*Lx;
        y0=.25*Ly;
        sigma0=(Lx*Ly)/500;
        x1=x0;
        y1=.75*Ly;
        sigma1=(Lx*Ly)/500;
        fh = @(x,y) (x/Lx).*(1-(x/Lx)).*(y/Ly).*(1-(y/Ly)).*(exp(-((x-x0).^2+(y-y0).^2)/sigma0)...
            +exp(-((x-x1).^2+(y-y1).^2)/sigma1));
    otherwise
        error('unknown function type %s in %s',function_type,mfilename)
end
% plotting to check the function
x=linspace(0,Lx,100);
y=linspace(0,Ly,100);

[X,Y]=meshgrid(x,y);
Z=zeros(length(x),length(y));
for i=1:length(x)
    for j=1:length(y)
        Z(i,j)=fh(x(i),y(j));
    end
end
figure(99); surf(X,Y,Z'); xlabel('x'); ylabel('y');

% subdivide the domain with Nx-1, Ny-1 cut lines (excluding the domain
% boundaries)
N_cutx=3;
N_cuty=3;

Ncol =N_cutx+1; Nrow =N_cuty+1;
Nptsx=N_cutx+2; Nptsy=N_cuty+2;
x=linspace(0,Lx,Nptsx);
y=linspace(0,Ly,Nptsx);

F = compute_integral_2d(fh,x,y);
check = integral2(fh,0,Lx,0,Ly) - sum(sum(F));
if abs(check)>sqrt(eps)
    error('integral error %g in %s',check,mfilename)
end
load_imbalance = max(max(F))/(sum(sum(F))/Nrow/Ncol);
fprintf('Initial load imbalance %g, max/min=%g \n',load_imbalance,max(max(F))/min(min(F)));

counter=1;
while load_imbalance>1.0001
    % fixed i in F(i,j) means values for column i
    Fcol=sum(F,2); % access as Fcol(i). column metric means x varies
    Frow=sum(F,1)'; % access as Frow(j). row    metric means y varies
    
    % F per row/column
    x_new = compute_new_cut_lines2(Fcol,x);
    y_new = compute_new_cut_lines2(Frow,y);
    % plot
    figure(1); clf; 
    surf(X,Y,Z'); xlabel('x'); ylabel('y');
    hold on;
    view(0,90);
    zzz=max(max(Z))*ones(2,1)*1.2;
    for i=2:length(x)-1
        line([x_new(i) x_new(i)],[y(1) y(end)],zzz);
    end
    for j=2:length(y)-1
        line([x(1) x(end)],[y_new(j) y_new(j)],zzz);
    end
    [x_new' y_new']
    x=x_new;
    y=y_new;
    % re-compute metric
    F = compute_integral_2d(fh,x_new,y_new);
    load_imbalance = max(max(F))/(sum(sum(F))/Nrow/Ncol);
    fprintf('Iteration %d, load imbalance %g, max/min=%g \n',counter,load_imbalance,max(max(F))/min(min(F)));
    counter=counter+1;
    pause
    if counter>10
        fprintf('not converged \n');
        break
    end
end






% % F_new=F_new/sum(sum(F_new));
% % fixed i in F(i,j) means values for column i
% Fcol_new=sum(F_new,2); % access as Fcol(i). column metric means x varies
% Frow_new=sum(F_new,1)'; % access as Frow(j). row    metric means y varies
% 
% [Frow Frow_new]
% [sum(Frow) sum(Frow_new)]
% [Fcol Fcol_new]
% [sum(Fcol) sum(Fcol_new)]

