function  [current,next,stop_sweep]=resolve_sweep_conflicts(diG,order,corners,current,next,nx,ny)

% get list of current nodes
nodes_only = current(:,1);
% get their unique list
[unique_nodes,ia,ic]=unique(nodes_only);
% a=[1 2 3 4 4 3 2 5]
% [c,ia,ic]=unique(a);ia=ia';ic=ic';
% c =     1     2     3     4     5
% ia =    1     2     3     4     8
% ic =     1     2     3     4     4     3     2     5

% if no conflict, exit this routine right away
if length(nodes_only)==length(unique_nodes)
    stop_sweep=false;
    return
end
stop_sweep=false;

% make list of nodes that have no conflict
no_conflict=zeros(0,3);
conflict=[];
conflicted=0;
for k=1:length(unique_nodes)
    ind = find(nodes_only == unique_nodes(k));
    if isempty(ind)
        error('ind should not be empty in %s',mfilename);
    end
    if length(ind)==1
        no_conflict = [ no_conflict; current(ind,:)];
    else
        conflicted = conflicted + 1;
        conflict{conflicted}=zeros(length(ind),3);
        for n=1:length(ind)
            conflict{conflicted}(n,:) = current(ind(n),:);
        end
    end
end

% no_conflict
% conflict

to_do =zeros(0,3);
to_lag=zeros(0,3);
option='old';
% resolve conflicts node by node
switch option
    case 'old'
        % using quadrant priority
        [to_do,to_lag,conflict,stop_sweep] = quadrant_priority(to_do,to_lag,conflict,nx,ny);
        %  then using quadrant priority, if needed
        if length(conflict)>0 && ~stop_sweep
            [to_do,to_lag,conflict,stop_sweep] = depth_of_graph(diG,order,corners,to_do,to_lag,conflict,nx,ny);
        end
    case 'depth_of_graph'
        [to_do,to_lag,conflict,stop_sweep] = depth_of_graph(diG,order,corners,to_do,to_lag,conflict,nx,ny);
end
% assign no_conflict and resolved conflicts to current
for k=1:length(to_do(:,1))
    no_conflict = [no_conflict; to_do(k,:)];
end
current=no_conflict;

% add lagged conflicts to next buffer
for k=1:length(to_lag(:,1))
    next = [next; to_lag(k,:)];
end

if length(conflict)>0
    stop_sweep=true;
end

return





