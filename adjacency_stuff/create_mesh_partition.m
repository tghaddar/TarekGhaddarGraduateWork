function [X,Y,nx,ny]=create_mesh_partition(n_cutx,n_cuty,partition_type)

% number of subsets per dimension
nx = n_cutx + 1;
ny = n_cuty + 1;

switch lower(partition_type)
    case 'random'
        % cut lines in x
        X = [0 sort(rand(1,n_cutx)) 1];
        % cut lines in y, per column
        Y=zeros(nx,n_cuty+2);
        for i=1:nx
            Y(i,:) = [0 sort(rand(1,n_cuty)) 1];
        end
    case 'regular'
        X = linspace(0,1,n_cutx+2);
        tmp = linspace(0,1,n_cuty+2);
        for i=1:nx
            Y(i,:) = tmp;
        end
    case 'mild_random'
        X = linspace(0,1,n_cutx+2);
        tmp = linspace(0,1,n_cuty+2);
        dy=tmp(2)-tmp(1);
        percent=0.1;
        for i=1:nx
            Y(i,:) = tmp + dy*[0 (2*rand(1,n_cuty)-1) 0]*percent;;
        end
    case 'worst'
        X = linspace(0,1,n_cutx+2);
        tmp_1 = [ linspace(0,0.25,n_cuty+1) 1];
        tmp_2 = [ 0 linspace(0.75,1,n_cuty+1)];
        for i=1:2:nx
            Y(i,:) = tmp_1;
        end
        for i=2:2:nx
            Y(i,:) = tmp_2;
        end
    case 'debug_random'
        load debug_cut.mat;
    case 'debug_random1818'
        load debug1818.mat;
    case 'mild_random1818'
        load mild1818.mat;
    case 'debug_regular'
        load debug_reg_cut.mat;
    otherwise
        error('unknown mesh partition in %s',mfilename)
end
