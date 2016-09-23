close all; clear all; clc;

do_random=true;

% number of cut lines
n_cutx = 3;
n_cuty = 3;

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

% adjacency matrix
n=nx*ny;
A = zeros(n,n);

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

% symmetric portion
A = A + A';
figure(1)
spy(A)

% plot the graph
figure(2)
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
gplot(A,Coordinates,'r-s');
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
    
% useful links:
% http://stackoverflow.com/questions/27339909/how-to-automatically-create-coordinates-when-graphing-adjacency-matrix-using-mat
% http://stackoverflow.com/questions/3277541/construct-adjacency-matrix-in-matlab
% 
% 
