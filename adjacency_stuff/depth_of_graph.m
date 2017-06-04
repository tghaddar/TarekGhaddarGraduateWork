function [to_do,to_lag,conflict_,stop_sweep] = depth_of_graph(diG,order,corners,to_do,to_lag,conflict,nx,ny)

if nx>=ny
    % further quadrant priority
    q_prio = [1 2 3 4];
    q_prio = [1 4 2 3];
else
    q_prio = [1 3 2 4];
    q_prio = [1 4 3 2];
end
opposite_quadrant=[4 3 2 1];
stop_sweep=false;
conflicted=length(conflict);

for k=1:conflicted
    
    n=length(conflict{k}(:,1));
%     if n~=2
%         conflict{k}
%         warning('only resolving conflicts pair-by-pair for now; in %s',mfilename);
% %         stop_sweep=true;
% %         break
%     end
    
    % quadrant IDs
    %  2---4
    %  |   |
    %  1---3
    q=zeros(n,1);
    for i=1:n
        q(i)=conflict{k}(i,2);
    end
    if length(unique(q))~=n
        for i=1:n
            conflict{k}(i,:)
        end
        error('same node, same quadrant, it cannot be due to as. Issue in %s',mfilename);
    end
    
    id=zeros(n,1);
    beg_=zeros(n,1);
    end_=zeros(n,1);
    c=zeros(n,1);
    dog=zeros(n,1);
    for i=1:n
        % real node IDs
        id(i) = conflict{k}(i,1);
        % get their IDs in their respective diG
        beg_(i)=find(order{q(i)}==id(i));
        % get real corner IDs in opposite direction
        c(i)=corners(opposite_quadrant(q(i)));
        % get their IDs in their respective diG
        end_(i)=find(order{q(i)}==c(i));
        % compute depth of graph
        dog(i)=graphshortestpath(adjacency(diG{q(i)}),beg_(i),end_(i));
    end
    if length(unique(id))~=1
        id
        error('nodes should have the same IDs, in %s',mfilename);
    end

    if any(isinf(dog))
        dog
        for i=1:n
            if isinf(dog(i))
                error('Cannot go from node %d to node %d in quadrant %d',beg_(i),end_(i),q(i));
            end
        end
    end
    
    [sorted_dog,ind_sort]=sort(dog,'descend');
    ind=find(sorted_dog==sorted_dog(1));
    if length(ind)==1
        i_to_do =ind_sort(1);
        i_to_lag=ind_sort(2:end);
    else
        qq=q(ind_sort(ind));
        for j=1:length(ind)
            indj(j)=find(q_prio==qq(j));
        end
        [~,best]=min(indj);
        i_to_do=ind_sort(ind(best));
        i_to_lag=[ind_sort(1:ind(best)-1) ind_sort(ind(best)+1:end)];
        % clean up before next conflict
        clear qq indj 
    end
    
    % apply
    if ~isempty(i_to_do)
        to_do =[to_do ; conflict{k}(i_to_do ,:)];
        to_lag=[to_lag; conflict{k}(i_to_lag,:)];
        conflict{k}=[];
    end

    % move on to the next conflict to be resolved using this rule
end

% completed remove empty cells
kk=0;conflict_=[];
for k=1:conflicted
    if ~isempty(conflict{k})
        kk=kk+1;
        conflict_{kk}=conflict{k};
    end
end