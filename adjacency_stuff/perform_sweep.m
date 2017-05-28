function result = perform_sweep(diG,order,X,Y,nx,ny,do_plot_sweep)

for quad=1:4
    
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
            starting_node = id; % ord(ind);
            % %         [dist, path, pred] = graphshortestpath(adj, starting_node);
            % %         n_stages_serial = max(dist) + 1;
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
    pred = zeros(siz_,how_many);
    succ = zeros(siz_,how_many);
    for k=1:n_nodes
        node_ids = predecessors(dg,k);
        if ~isempty(node_ids)
            pred(k,1:length(node_ids)) = node_ids;
        end
        node_ids = successors(dg,k);
        if ~isempty(node_ids)
            succ(k,1:length(node_ids)) = node_ids;
        end
    end
    [~,aux]=size(pred);
    if how_many<aux
        warning('need to increase how many (%d) to %d b/c pred; in %s',how_many,aux,mfilename);
        how_many=aux;
    end
    
    % create buffer of work to do
    my_buffer = ord;
    % initialize the nodes for which there is work to do
    current_nodes = starting_node;
    % remove current node from buffer
    ind = find(my_buffer==current_nodes(1));
    if isempty(ind)
        error('the current node %d should not yet have been worked on; in %s',current_nodes(1),mfilename);
    else
        my_buffer(ind)=[];
    end
    n_stages = 1;
    wave{n_stages} = current_nodes;
    % make a buffer for nodes for which some predecessors have been completed
    % but not all
    not_ready=[];
    
    while ~isempty(my_buffer)
        
        fprintf('Stage #: %d \n',n_stages+1);
        
        % remove current nodes from list of predecessors (work by column is
        % quicker)
        for k=1:length(current_nodes)
            [indi,indj] = find(pred==current_nodes(k));
            for i=1:length(indi)
                pred(indi(i),indj(i))=0;
            end
        end
        
        % get next nodes by looking at the successors
        next_nodes = not_ready;
        for k=1:length(current_nodes)
            ind = find(succ(current_nodes(k),:)>0);
            next_nodes = [next_nodes succ(current_nodes(k),ind)];
        end
        next_nodes = unique(next_nodes);
        %     fmt=[repmat('%d ',1,length(next_nodes)) '\n'];
        %     fprintf('Potential next nodes: %s',fmt,next_nodes);
        
        % check that these next nodes are free
        for k=length(next_nodes):-1:1
            ind = find(pred(next_nodes(k),:)>0, 1);
            if ~isempty(ind)
                % that node still has a pred that hasn't been worked on
                not_ready = [not_ready next_nodes(k)];
                % we do the loop from the end because of this line            next_nodes(k)=[];
                next_nodes(k)=[];
            end
        end
        not_ready = unique(not_ready);
        % assign them a new name
        current_nodes=next_nodes;
        
        % remove the free nodes from the buffer
        for k=1:length(current_nodes)
            ind = find(my_buffer==current_nodes(k));
            if isempty(ind)
                error('the current node %d should not yet have been worked on; in %s',current_nodes(k),mfilename);
            else
                my_buffer(ind)=[];
            end
            % also remove the nodes that have been completed from the not_ready buffer
            ind = find(not_ready==current_nodes(k));
            if ~isempty(ind)
                not_ready(ind)=[];
            end
        end
        
        % increment the stage count
        n_stages = n_stages + 1;
        wave{n_stages} = current_nodes;
        
    end
    
    if do_plot_sweep
        plot_sweep(wave,ord,X,Y,nx,ny,quad);
    end
    % save all results
    result.n_stages(quad) = n_stages;
    result.wave{quad} = wave;
    clear wave;
end