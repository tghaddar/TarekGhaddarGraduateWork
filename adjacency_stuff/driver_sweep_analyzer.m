clear variables; close all; clc;

% common options to be used
inp.plot_mesh = false;
inp.plot_dag = false;
inp.do_plot_sweep = false;

inp.save_case = true;
inp.save_ID = 0;

inp.conflict_option = 'dog';

% list of parameters that vary
testname = 'jcr_v1';
switch testname
    case 'jcr_v1'
        cutx = [3 5 10 20 50];
        as   = [ 1 3 5 10 20];
        k=0;
        k=k+1; partition_type{k} = 'regular';
        k=k+1; partition_type{k} = 'mild_random';
        k=k+1; partition_type{k} = 'random';
        k=k+1; partition_type{k} = 'worst';
    case 'jcr_v2'
        cutx = [3 5 10 20 50 100 150 200];
        as   = [1];
        k=0;
        k=k+1; partition_type{k} = 'regular';
        k=k+1; partition_type{k} = 'mild_random';
        k=k+1; partition_type{k} = 'random';
        k=k+1; partition_type{k} = 'worst';
    case 'jcr_v3'
        cutx = [200];
        as   = [1];
        k=0;
        k=k+1; partition_type{k} = 'worst';
    case 'jcr_v4a'
        cutx = 10;
        as   = [1];
        k=0;
        k=k+1; partition_type{k} = 'regular';
    case 'jcr_v4b'
        cutx = 20*ones(10,1); % 20 realizations with 10 cuts
        as   = [1];
        k=0;
        k=k+1; partition_type{k} = 'mild_random';
        k=k+1; partition_type{k} = 'random';
    case 'jcr_v5a'
        cutx = 50;
        as   = [1];
        k=0;
        k=k+1; partition_type{k} = 'regular';
    case 'jcr_v5b'
        cutx = 50*ones(20,1); % 20 realizations with 50 cuts
        as   = [1];
        k=0;
        k=k+1; partition_type{k} = 'mild_random';
        k=k+1; partition_type{k} = 'random';
    case 'jcr_v6a'
        cutx = [300];
        as   = [1];
        k=0;
        k=k+1; partition_type{k} = 'worst';
        k=k+1; partition_type{k} = 'regular';
        k=k+1; partition_type{k} = 'mild_random';
        k=k+1; partition_type{k} = 'random';
    case 'jcr_v6b'
        cutx = [400];
        as   = [1];
        k=0;
        k=k+1; partition_type{k} = 'regular';
        k=k+1; partition_type{k} = 'mild_random';
        k=k+1; partition_type{k} = 'random';
        k=k+1; partition_type{k} = 'worst';
    case 'jcr_v7a'
        cutx = [1 2 3 4 5 10 20];
        as   = [1 3 5];
        k=0;
        k=k+1; partition_type{k} = 'regular';
    case 'jcr_v7b'
        cutx = kron([1 2 3 4 5 10 20],ones(1,20)); % 20 realizations with 10 cuts
        as   = [1 3 5];
        k=0;
        k=k+1; partition_type{k} = 'mild_random';
        k=k+1; partition_type{k} = 'random';
end

% cutx = [ 3 ];
% as   =[ 1 3 ];
% k=1;   partition_type{k} = 'regular';

if strcmp(inp.conflict_option,'both')
    n_stages=zeros(length(partition_type),length(cutx),length(as),2);
else
    n_stages=zeros(length(partition_type),length(cutx),length(as),1);
end

for i_partition=1:length(partition_type)
    inp.partition_type = partition_type{i_partition};
    for i_cutx=1:length(cutx)
        inp.n_cutx = cutx(i_cutx);
        inp.n_cuty = inp.n_cutx;
        for i_as=1:length(as)
            inp.n_angle_sets = as(i_as);
            % call analyzer
            fprintf('Working on %s, cut=%d, as=%d\n',partition_type{i_partition},cutx(i_cutx),as(i_as));
            if strcmp(inp.conflict_option,'both')
                [out_pdt,out_dog]=sweep_analyzer_fun(inp);
                n_stages(i_partition,i_cutx,i_as,1)=out_pdt.n_stages;
                n_stages(i_partition,i_cutx,i_as,2)=out_dog.n_stages;
            else
                [out]=sweep_analyzer_fun(inp);
                n_stages(i_partition,i_cutx,i_as,1)=out.n_stages;
            end
            fprintf('\n\n');
        end
    end
end

file_save =sprintf('sweep_analysis_%s.mat',testname);
save(file_save,'partition_type','cutx','as','n_stages')

disp('done');
