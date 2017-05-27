function [X,Y,nx,ny]=create_mesh_partition(n_cutx,n_cuty,do_random)

% number of subsets per dimension
nx = n_cutx + 1;
ny = n_cuty + 1;

if do_random
    % cut lines in x
    X = [0 sort(rand(1,n_cutx)) 1];
    % cut lines in y, per column
    Y=zeros(nx,n_cuty+2);
    for i=1:nx
        Y(i,:) = [0 sort(rand(1,n_cuty)) 1];
    end
else
    X = linspace(0,1,n_cutx+2);
    tmp = linspace(0,1,n_cuty+2);
    for i=1:nx
        Y(i,:) = tmp;
    end
end

