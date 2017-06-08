clear variables; close all; clc;

load sweep_analysis_jcr_v1.mat
n_stages(:,:,:,1)=[];
len=length(n_stages(1,1,1,:));
switch len
    case 1
        dog_only=true;
    case 2
        dog_only=false;
    otherwise
        error('only 2 conflict resolution options max: %d in %s',len,mfilename);
end

% one figure per partition type, nstages versus ncuts for various angle
% sets
for i_partition=1:length(partition_type)
    figure(i_partition); hold all
    leg=char('dummy');
    name = partition_type{i_partition};
    %     name(regexp(name,'_'))=[] ;
    name = regexprep(name,'_',' ');
    for i_as=1:length(as)
        if ~dog_only
            plot(cutx,n_stages(i_partition,:,i_as,1),'-+'); 
            leg=char(leg,sprintf('%s, as=%d, pdt',name,as(i_as)));
        end
        plot(cutx,n_stages(i_partition,:,i_as,end),'-o'); 
        leg=char(leg,sprintf('%s, as=%d, dog',name,as(i_as)));
    end
    title(sprintf('Partition %s',name));
    ylabel('n stages');
    xlabel('cuts in x/y');
    grid on
    leg(1,:)=[];
    legend(leg,'Location','northwest')
end

% one figure per angle set, nstages versus ncuts for various partition
% types
for i_as=1:length(as)
    figure(10+i_as); hold all
    leg=char('dummy');
    for i_partition=1:length(partition_type)
        name = partition_type{i_partition};
        name = regexprep(name,'_',' ');
        if ~dog_only
            plot(cutx,n_stages(i_partition,:,i_as,1),'-+');
            leg=char(leg,sprintf('%s, as=%d, pdt',name,as(i_as)));
        end
        plot(cutx,n_stages(i_partition,:,i_as,end),'-o');
        leg=char(leg,sprintf('%s, as=%d, dog',name,as(i_as)));
    end
    title(sprintf('Angle set %d',as(i_as)));
    ylabel('n stages');
    xlabel('cuts in x/y');
    grid on
    leg(1,:)=[];
    legend(leg,'Location','northwest')
end


disp('done');
