clear variables; close all; clc;

% common options to be used
inp.plot_mesh = false;
inp.plot_dag = false;
inp.do_plot_sweep = false;

inp.save_case = false;
inp.save_ID = 0;

inp.conflict_option = 'both';

% list of parameters that vary
cutx = [3 5 10 20 50];
as   = [ 1 3 5 10 20];
k=0;
k=k+1;   partition_type{k} = 'regular';
k=k+1; partition_type{k} = 'random';
k=k+1; partition_type{k} = 'mild_random';
k=k+1; partition_type{k} = 'worst';

% cutx = [ 3 ];
% as   =[ 1 3 ];
% k=1;   partition_type{k} = 'regular';

n_stages=zeros(length(partition_type),length(cutx),length(as),2);

for i_partition=1:length(partition_type)
    inp.partition_type = partition_type{i_partition};
    for i_cutx=1:length(cutx)
        inp.n_cutx = cutx(i_cutx);
        inp.n_cuty = inp.n_cutx;
        for i_as=1:length(as)
            inp.n_angle_sets = as(i_as);
            % call analyzer
            fprintf('Working on %s, cut=%d, as=%d\n',partition_type{i_partition},cutx(i_cutx),as(i_as));
            [out_pdt,out_dog]=sweep_analyzer_fun(inp);
            n_stages(i_partition,i_cutx,i_as,1)=out_pdt.n_stages;
            n_stages(i_partition,i_cutx,i_as,2)=out_dog.n_stages;
            fprintf('\n\n');
        end
    end
end

save sweep_analysis.mat partition_type cutx as n_stages

disp('done');
