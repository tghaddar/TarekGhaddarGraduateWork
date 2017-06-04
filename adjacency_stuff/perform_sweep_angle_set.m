function result = perform_sweep_angle_set(diG,order,corners,X,Y,nx,ny,n_angle_sets,do_plot_sweep)

n_quad=4;
%%% init phase for all quadrants and all angle sets
for quad=1:n_quad
    
    % select one quadrant
    dg=diG{quad};
    ord=order{quad};
    % % re-create current adjacency mat
    % adj= adjacency(dg);
    
    % compute in degree
    in_degree  = indegree(dg);
    id = find(in_degree==0);
    switch length(id)
        case 1
            % normal, one corner only to begin with
            starting_node(quad) = ord(id);
        case 0
            error('bug or cyclic graph in %s',mfilename);
        otherwise
            error('too many starting nodes in %s',mfilename);
    end
    % compute out degree
    out_degree = outdegree(dg);
    % check dimensions
    siz_ = length(in_degree);
    n_nodes=numnodes(dg);
    if siz_ ~= nx*ny || siz_ ~= n_nodes
        error('inconsistency in the number of nodes in %s',mfilename);
    end
    how_many = max (max(in_degree),max(out_degree) );
    % predecessors and successors arrays
    pred{quad} = zeros(siz_,how_many,n_angle_sets);
    succ{quad} = zeros(siz_,how_many,n_angle_sets);
    for k=1:n_nodes
        node_ids = ord(predecessors(dg,k));
        if ~isempty(node_ids)
            for as=1:n_angle_sets
                pred{quad}(ord(k),1:length(node_ids),as) = node_ids; %ord(node_ids);
            end
        end
        node_ids = ord(successors(dg,k));
        if ~isempty(node_ids)
            for as=1:n_angle_sets
                succ{quad}(ord(k),1:length(node_ids),as) = node_ids; %ord(node_ids);
            end
        end
    end
    
%     % create buffer of tasks, including info about quadrant and anglet set IDs
%     for as=1:n_angle_sets
%         my_buffer{quad,as}=order{quad};
%     end
    
end

n_tasks = (nx*ny)*n_quad*n_angle_sets;

% initialize the nodes for which there is work to do
% nodes | quadrant | angle set
qq=linspace(1,n_quad,n_quad);
current_nodes(1:n_quad,1:3) = [starting_node' qq' ones(n_quad,1)];
% remove current nodes from tasks
n_tasks = n_tasks-n_quad;

n_stages = 1;
wave{n_stages} = current_nodes;
% make a buffer for nodes for which some predecessors have been completed
% but not all
if n_angle_sets==1
    potentially_next=zeros(0,3);
else
    potentially_next(1:n_quad,1:3)=[starting_node' qq' 2*ones(n_quad,1)];
end

while n_tasks>0
    
    fprintf('Stage #: %d \n',n_stages+1);
    
    % remove current nodes from list of predecessors (work by column is
    % quicker)
    for q=1:n_quad
        indq= current_nodes(:,2)==q;
        % current nodes for given quadrant
        cn_q=current_nodes(indq,:);
        for as=1:n_angle_sets
            indas= cn_q(:,3)==as;
            % current nodes for given as
            cn_as=cn_q(indas,:);
            for k=1:length(cn_as(:,1))
                [indi,indj] = find(pred{q}(:,:,as)==cn_as(k,1));
                for i=1:length(indi)
                    pred{q}(indi(i),indj(i),as)=0;
                end
            end
        end
    end
    
    % get next nodes by looking at the successors
    next_nodes = potentially_next;
    for k=1:length(current_nodes(:,1))
        quad=current_nodes(k,2);
        as  =current_nodes(k,3);
        ind = find(succ{quad}(current_nodes(k,1),:,as)>0);
        my_succ = succ{quad}(current_nodes(k,1),ind,as);
        my_quad = quad*ones(length(ind),1);
        my_as   = as*ones(length(ind),1);
        next_nodes = [next_nodes; [my_succ' my_quad my_as] ];
    end
    next_nodes = unique(next_nodes,'rows');
    
    % remove multiple as's for the list of nexy nodes
    % a=[1 2 3 4 4 3 2 5]
    % [c,ia,ic]=unique(a);ia=ia';ic=ic';
    % c =     1     2     3     4     5
    % ia =    1     2     3     4     8
    % ic =     1     2     3     4     4     3     2     5
    [c,iu,ic]=unique(next_nodes(:,1:2),'rows');
    to_keep=zeros(0,3);
    for k=1:length(iu)
        ind=find(ic==k);
        if length(ind)>1
            [~,lowest_as]=min(next_nodes(ind,3));
            to_keep = [to_keep ; next_nodes(ind(lowest_as),:)];
        elseif length(ind)==1
            to_keep = [to_keep ; next_nodes(ind(1),:)];
        else
            error('ind must be >=1 in %s',mfilename);
        end
    end
    next_nodes=to_keep;
    
    % check that these next nodes are free
    for k=length(next_nodes(:,1)):-1:1
        quad=next_nodes(k,2);
        as  =next_nodes(k,3);
        ind = find(pred{quad}(next_nodes(k),:,as)>0, 1);
        if ~isempty(ind)
            % that node still has a pred that hasn't been worked on
            potentially_next = [potentially_next; next_nodes(k,:)];
            % we do the loop from the end because of this line: next_nodes(k,:)=[];
            next_nodes(k,:)=[];
        end
    end
    potentially_next = unique(potentially_next,'rows');
    % assign them a new name
    current_nodes=next_nodes;
    
    % resolve conflicts (nodes doing more than one task)
    [current_nodes,potentially_next,stop_sweep]=resolve_sweep_conflicts(diG,order,corners,...
        current_nodes,potentially_next,nx,ny);
    
    % remove current nodes from tasks
    n_tasks = n_tasks - length(current_nodes(:,1));

    % also remove the nodes that have been completed from the potentially_next buffer
    for k=1:length(current_nodes(:,1))
        ind = find(ismember(potentially_next,current_nodes(k,:),'rows'));
        if ~isempty(ind)
            potentially_next(ind,:)=[];
        end
    end
    
    % add the next angle set in the list of potential next nodes
    for k=1:length(current_nodes(:,1))
        as=current_nodes(k,3);
        if as<n_angle_sets
            % increment the as by 1
            potentially_next(end+1,:)=[current_nodes(k,1:2) (as+1)];
        end
    end    

% %     for k=1:length(current_nodes(:,1))
% %         ind = find(my_buffer==current_nodes(k));
% %         if isempty(ind)
% %             error('the current node %d should not yet have been worked on; in %s',current_nodes(k),mfilename);
% %         else
% %             my_buffer(ind)=[];
% %         end
% %         % also remove the nodes that have been completed from the not_ready buffer
% %         ind = find(not_ready==current_nodes(k));
% %         if ~isempty(ind)
% %             not_ready(ind)=[];
% %         end
% %     end
    
    if stop_sweep
        n_tasks=0;
        warning('sweep did not conclude');
    else
        % increment the stage count
        n_stages = n_stages + 1;
        wave{n_stages} = current_nodes;
    end
    
end

if do_plot_sweep
    plot_sweep_as(wave,n_angle_sets,X,Y,nx,ny);
end
% save all results
result.n_stages= n_stages;
result.wave = wave;
