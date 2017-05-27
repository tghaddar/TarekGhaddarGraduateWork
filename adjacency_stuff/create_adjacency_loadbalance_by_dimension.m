close all; clear variables; % clc;
debug=true;

% number of cut lines
n_cutx = 3;
n_cuty = 3;
% random cuts
do_random=false;

[X,Y,ny,nx] = create_mesh_partition(n_cutx,n_cuty,do_random);
if debug
    clear X Y nx ny;
    if do_random
        load debug_cut.mat;
    else
        load debug_reg_cut.mat;
    end
end

plot_mesh=true;
plot_dag=true;
[diG,order,corners]=create_adjacency_matrices(X,Y,nx,ny,plot_mesh,plot_dag);

do_plot_sweep=true;
perform_sweep(diG,order,X,Y,nx,ny,do_plot_sweep);

error('stopping here')

G_spmat=sparse(A_sym);
DG_spmat=sparse(A_nonsym);
BioG=biograph(DG_spmat); % h = view(BG);

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
