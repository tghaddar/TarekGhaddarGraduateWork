function varargout=sweep_analyzer_fun(varargin)

if nargin==0
    
    close all; clear variables; clc;
    
    % number of cut lines
    n_cutx = 3;
    n_cuty = 3;
    % partition_type='debug_regular';
    % partition_type='debug_random';
    % partition_type='debug_random1818';
    % partition_type='regular';
    partition_type='random';
    % partition_type='mild_random';
    % partition_type='mild_random1818';
    % partition_type='worst';
    
    plot_mesh=false;
    plot_dag=false;
    do_plot_sweep=true;

    n_angle_sets=1;
    conflict_option='old';
    
    save_case = false;
    save_ID = 0;

else
        %%% get input from varargin
    inp=varargin{1};
    
    n_cutx = inp.n_cutx;
    n_cuty = inp.n_cuty;
    partition_type = inp.partition_type;
    
    n_angle_sets = inp.n_angle_sets;
    
    conflict_option = inp.conflict_option;
    
    plot_mesh = inp.plot_mesh;
    plot_dag = inp.plot_dag;
    do_plot_sweep = inp.do_plot_sweep;
    
    save_case = inp.save_case;
    save_ID = inp.save_ID;
    

end
    
% create partition typpe
[X,Y,nx,ny] = create_mesh_partition(n_cutx,n_cuty,partition_type);
if ny>nx
    warning('we assume Px >= Py for angle priorities, so we need nx>=ny');
end

% create adjacency matrix
[diG,order,corners]=create_adjacency_matrices(X,Y,nx,ny,plot_mesh,plot_dag);

% 1-direction sweep
% rez1 = perform_sweep(diG,order,X,Y,nx,ny,do_plot_sweep);

switch conflict_option
    case {'old','dog'}
        rez = perform_sweep_angle_set(diG,order,corners,X,Y,nx,ny,n_angle_sets,conflict_option,...
            partition_type,do_plot_sweep);
        varargout{1}=rez;
        if nargout~=1
            error('nargout should be =1, it is %d, in %s',nargout,mfilename);
        end
    case 'both'
        rez_old = perform_sweep_angle_set(diG,order,corners,X,Y,nx,ny,n_angle_sets,'old',...
            partition_type,do_plot_sweep);
        varargout{1}=rez_old;
        rez_dog = perform_sweep_angle_set(diG,order,corners,X,Y,nx,ny,n_angle_sets,'dog',...
            partition_type,do_plot_sweep);
        varargout{2}=rez_dog;
        if nargout~=2
            error('nargout should be =2, it is %d, in %s',nargout,mfilename);
        end
end

if save_case
    filename_mat=sprintf('%s_%dx%d_as%d_v%d.mat',partition_type,nx,ny,n_angle_sets,save_ID);
    save(filename_mat, 'X', 'Y', 'nx', 'ny', 'rez_old', 'rez_dog');
end


return


%%%%%%%% old stuff below. still keep it for now
error('stopping here')

G_spmat =sparse(A_sym);
DG_spmat=sparse(A_nonsym);
BioG=biograph(DG_spmat); % h = view(BioG);

% disc is a vector of node indices in the order in which they are discovered. 
% pred is a vector of predecessor node indices (listed in the order of the node indices) 
% of the resulting spanning tree. 
% closed is a vector of node indices in the order in which they are closed.
[disc, pred, closed] = graphtraverse(DG_spmat, 1);
[disc_, pred_, closed_] = graphtraverse( G_spmat, 1, 'Directed', false );

% dist are the N distances from the source to every node (using Infs for 
% nonreachable nodes and 0 for the source node). path contains the winning 
% paths to every node. pred contains the predecessor nodes of the winning paths.
[dist, path, pred] = graphshortestpath(DG_spmat, 1);

% finds the shortest paths between every pair of nodes in the graph 
% represented by matrix G, using Johnson's algorithm. Input G is an N-by-N 
% sparse matrix that represents a graph. Nonzero entries in matrix G 
% represent the weights of the edges.
[dist] = graphallshortestpaths(DG_spmat);
[dist] = graphallshortestpaths(G_spmat);

%%%%%%%%%% same ting but using a BG ...
% dist are the N distances from the source to every node (using Infs for 
% nonreachable nodes and 0 for the source node). 
% path contains the winning paths to every node. 
% pred contains the predecessor nodes of the winning paths.
[dist, path, pred] = shortestpath(BioG, 1);

% Output dist is an N-by-N matrix where dist(S,T) is the distance of the 
% shortest path from source node S to target node T. 
[dist] = allshortestpaths(BioG)



% useful links:
% http://stackoverflow.com/questions/27339909/how-to-automatically-create-coordinates-when-graphing-adjacency-matrix-using-mat
% http://stackoverflow.com/questions/3277541/construct-adjacency-matrix-in-matlab
% 
% 
