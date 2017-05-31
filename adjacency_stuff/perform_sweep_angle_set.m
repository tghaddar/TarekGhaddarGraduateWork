function result = perform_sweep_angle_set(diG,order,X,Y,nx,ny,n_angle_sets,do_plot_sweep)

n_quad=4;
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
            pred{quad}(ord(k),1:length(node_ids),:) = node_ids; %ord(node_ids);
        end
        node_ids = ord(successors(dg,k));
        if ~isempty(node_ids)
            succ{quad}(ord(k),1:length(node_ids),:) = node_ids; %ord(node_ids);
        end
    end
    
    % create buffer of tasks, including info about quadrant and anglet set IDs
    for as=1:n_angle_sets
        my_buffer{quad,as}=order{quad};
    end
    
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
not_ready=zeros(n_angle_sets-1,3);
for as=2:n_angle_sets
    not_ready((as-1)*n_quad+1:as*n_quad,1:3)=[starting_node' qq' as*ones(n_quad,1)];
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
    next_nodes = not_ready;
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
    
    % check that these next nodes are free
    for k=length(next_nodes(:,1)):-1:1
        quad=next_nodes(k,2);
        as  =next_nodes(k,3);
        ind = find(pred{quad}(next_nodes(k),:,as)>0, 1);
        if ~isempty(ind)
            % that node still has a pred that hasn't been worked on
            not_ready = [not_ready; next_nodes(k,:)];
            % we do the loop from the end because of this line: next_nodes(k,:)=[];
            next_nodes(k,:)=[];
        end
    end
    not_ready = unique(not_ready,'rows');
    % assign them a new name
    current_nodes=next_nodes;
    
    % resolve conflicts (nodes doing more than one task)
    % TBD
    
    % remove current nodes from tasks
    n_tasks = n_tasks - length(current_nodes(:,1));

    % also remove the nodes that have been completed from the not_ready buffer
    for k=1:length(current_nodes(:,1))
        ind = find(ismember(not_ready,current_nodes(k,:),'rows'));
        if ~isempty(ind)
            not_ready(ind,:)=[];
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
    
    % increment the stage count
    n_stages = n_stages + 1;
    wave{n_stages} = current_nodes;
    
end

if do_plot_sweep
    plot_sweep_as(wave,order,X,Y,nx,ny);
end
% save all results
result.n_stages= n_stages;
result.wave = wave;
