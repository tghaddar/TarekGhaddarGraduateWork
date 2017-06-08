clear variables; close all; clc;

load sweep_analysis.mat

% one figure per partition type, nstages versus ncuts for various angle
% sets
for i_partition=1:length(partition_type)
    figure(i_partition); hold all
    leg=char('dummy');
    name = partition_type{i_partition};
    %     name(regexp(name,'_'))=[] ;
    name = regexprep(name,'_',' ');
    for i_as=1:length(as)
        plot(as,n_stages(i_partition,:,i_as,1),'-+'); 
        leg=char(leg,sprintf('%s, as=%d, pdt',name,as(i_as)));
        plot(as,n_stages(i_partition,:,i_as,2),'-o'); 
        leg=char(leg,sprintf('%s, as=%d, dog',name,as(i_as)));
    end
    title(sprintf('Partition %s',partition_type{i_partition}));
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
        plot(as,n_stages(i_partition,:,i_as,1),'-+');
        leg=char(leg,sprintf('%s, as=%d, pdt',name,as(i_as)));
        plot(as,n_stages(i_partition,:,i_as,2),'-o');
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
