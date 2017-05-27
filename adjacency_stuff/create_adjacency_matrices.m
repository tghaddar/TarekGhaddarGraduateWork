function [diGG,order,corners]=create_adjacency_matrices(X,Y,nx,ny,plot_mesh,plot_dag)

% adjacency matrix
n=nx*ny;
A = zeros(n,n);

% corner nodes of graph (needed for sweep by quadrant)
%   2---4
%   |   |
%   1---3
corners = zeros(4,1);
corners(1) = 1;
corners(2) = ny;
corners(3) = ny*(nx-1)+1;
corners(4) = n;

%%%% create adjacency matrix starting with corner #1
% loop over columns
for i=1:nx
    % beginning and end rows in A for this colum
    row_beg = (i-1)*ny+1;
    row_end = (i  )*ny  ;
    % up/down neighbors
    for row=row_beg:row_end
        % top neighbor
        if row~=row_end, A(row,row+1)=1; end
        % not need for bottom neighbors, we use transpose at the end
        % % bottom neighbor
        % if row~=row_beg, A(row,row-1)=1; end
    end
    
    % right neighbors, not needed for last column
    if i<nx
        % cut lines of the current column
        cut_lines_curr = Y(i,:);
        % cut lines of the right neighbor column
        cut_lines_neig = Y(i+1,:);
        for j=1:ny
            % current row in A matrix
            current_row = (i-1)*ny+j;
            % cut lines of interest
            Y1=cut_lines_curr(j);
            Y2=cut_lines_curr(j+1);
            ind = find( (cut_lines_neig>Y1-eps) & (cut_lines_neig<Y2+eps) );
            % TODO: need fuzzy logic
            % ind is the ID's of the cut lines of the neighbors
            if ~isempty(ind)
                connecting_subsets = [ind(1)-1 ind];
                if abs(Y1 - cut_lines_neig(ind(1)))< 1.5*eps
                    % remove the first subset
                    connecting_subsets(1) = [];
                end
                if abs(Y2 - cut_lines_neig(ind(end)))< 1.5*eps
                    % remove the last subset
                    connecting_subsets(end) = [];
                end
                % i*ny: to skip all ofthe subset in all columns from 1 to i included
                A(current_row,i*ny+connecting_subsets)=1;
            else
                % if ind is empty, then no cut lines of the neighboring
                % column intersect the given subset but that subset still
                % has a neighbor!!!
                ind1 = find(cut_lines_neig<Y1+eps);
                ind2 = find(cut_lines_neig>Y2-eps);
                if (ind1(end)-ind2(1)~=-1)
                    [i j]
                    ind1
                    ind2
                    warning('these 2 numbers must differ by -1 only');
                else
                    A(current_row,i*ny+ind1(end))=2;
                end
            end % ind is not empty
        end % loop over subset in given column
    end % we are not yet at the last column
    
end % loop of columns

% save the non-symmetric part
A_nonsym=A;
% save the symmetric portion
A_sym = A + A'; clear A;


figID=0;

if plot_mesh
    figID=figID+1;
    figure(figID);
    % pbaspect([2 1 1])
    
    % spy of the adjacency matrix (undirected)
    subplot(1,3,2);
    spy(A_sym);
    
    % plot the graph
    subplot(1,3,1);
    Coordinates = zeros(n,2);
    k=0;
    for i=1:nx
        xval = (X(i)+X(i+1))/2;
        for j=1:ny
            yval = (Y(i,j)+Y(i,j+1))/2;
            k=k+1;
            Coordinates(k,:)=[xval yval];
        end
    end
    gplot(A_sym,Coordinates,'r-s');
    hold on
    % superimpose the subset mesh
    for i=1:nx+1
        line([X(i),X(i)],[0,1])
        if i<=nx
            for j=1:ny+1
                line([X(i),X(i+1)],[Y(i,j),Y(i,j)])
            end
        end
    end
end

%%% construct the undirected graph based on A
%%% I do not think G is needed ...
if plot_mesh
    [row,col,~]=find(A_nonsym);
    G=graph(row,col);
    subplot(1,3,3);
    plot(G,'Layout','force'); %,'EdgeLabel',G.Edges.Weight);
    % [~,order] = sort(degree(G),'ascend')
end

%%% construct the directed graph based on A
diG = digraph(A_nonsym);
figure(3);
plot(diG,'Layout','force','EdgeLabel',diG.Edges.Weight);

%%% ordering for each quadrant
%   2---4
%   |   |
%   1---3
order{1}=linspace(1,n,n);
diGG{1}=diG;
order{2}=zeros(n,1);
k=0;
for i=1:nx
    for j=ny:-1:1
        ind=(i-1)*ny+j;
        k=k+1;
        order{2}(ind)=k;
    end
end
aux=reordernodes(diG,order{2}); diGG{2}=aux;
% figure(99); clf; plot(aux); pause
% ooo=toposort(G2,'Order','stable');
% G2=reordernodes(G2,ooo);
% figure(98); clf; plot(G2); pause
order{3}=zeros(n,1);
k=0;
for i=nx:-1:1
    for j=1:ny
        ind=(i-1)*ny+j;
        k=k+1;
        order{3}(ind)=k;
    end
end
aux=reordernodes(diG,order{3});  diGG{3}=aux;
% figure(99); clf; plot(aux); pause
order{4}=zeros(n,1);
k=0;
for i=nx:-1:1
    for j=ny:-1:1
        ind=(i-1)*ny+j;
        k=k+1;
        order{4}(ind)=k;
    end
end
aux=reordernodes(diG,order{4});  diGG{4}=aux;
% figure(99); clf; plot(aux); pause

if plot_dag
    figID=figID+1;
    figure(figID);
    for k=1:4
        subplot(2,2,k); spy(adjacency(diGG{k}));
    end
    figID=figID+1;
    figure(figID);
    for k=1:4
        subplot(2,2,k); plot(diGG{k});
    end
end


% check that the corners are correct (debug mode)
% for k=1:n
%     [disc,~,~]=graphtraverse(G_spmat,k,'Directed',true);
% end


