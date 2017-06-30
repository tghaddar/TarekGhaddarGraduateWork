close all; clc; clear variables;

% number of cut lines
n_cutx = 19;
n_cuty = 19;
% partition_type='debug_regular';
% partition_type='debug_random';
% partition_type='debug_random1818';
% partition_type='regular';
partition_type='random';
% partition_type='mild_random';
% partition_type='mild_random1818';
partition_type='worst';

% create partition typpe
[X,Y,nx,ny] = create_mesh_partition(n_cutx,n_cuty,partition_type);
if ny>nx
    warning('we assume Px >= Py for angle priorities, so we need nx>=ny');
end

plot_mesh=false;
plot_dag=false;
% create adjacency matrix
[diG,order,corners]=create_adjacency_matrices(X,Y,nx,ny,plot_mesh,plot_dag);

% 1-direction sweep
% rez1 = perform_sweep(diG,order,X,Y,nx,ny,do_plot_sweep);

do_plot_sweep=true;
n_angle_sets=1;
% conflict_option='old';
% rez_old = perform_sweep_angle_set(diG,order,corners,X,Y,nx,ny,n_angle_sets,conflict_option,...
%     partition_type,do_plot_sweep);
conflict_option='dog';
rez_dog = perform_sweep_angle_set(diG,order,corners,X,Y,nx,ny,n_angle_sets,conflict_option,...
    partition_type,do_plot_sweep);

save_case = false;
save_ID = 0;
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
