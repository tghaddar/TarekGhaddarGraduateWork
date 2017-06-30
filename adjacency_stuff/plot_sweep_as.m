function plot_sweep_as(wave,X,Y,nx,ny,partition_type,n_angle_sets,conflict_option)

filename_gif=sprintf('./sweep_pix/sweep_%s_%dx%d_as%d_%s.gif',partition_type,nx,ny,n_angle_sets,conflict_option);

filename_png=sprintf('./sweep_pix/sweep_%s_%dx%d_as%d_%s_',partition_type,nx,ny,n_angle_sets,conflict_option);

n_stages = length(wave);

figure(100);
% superimpose the subset mesh
for i=1:nx+1
    line([X(i),X(i)],[0,1])
    if i<=nx
        for j=1:ny+1
            line([X(i),X(i+1)],[Y(i,j),Y(i,j)])
        end
    end
end
% save figure data
h1=gcf;
objects=allchild(h1);
% colors for patches
col{1}='blue';
col{2}='black';
col{3}='green';
col{4}='red';
% create gradation of colors for angle sets
gradation=[0.25 0.75];
gradation2=linspace(0,1,n_angle_sets+1);

% loop through stages
for s=1:n_stages
    
    figure(101); clf;
    h2=gcf;
    copyobj(get(h1,'children'),h2);
    hold on;
    
    nodes = wave{s}(:,1);
    quad  = wave{s}(:,2);
    as    = wave{s}(:,3);
%     wave{s}
    % id = (i-1)*ny+j
    for k=1:length(nodes)
%         if quad(k)~=4
%             continue
%         end
        id = nodes(k); %order{quad(k)}(nodes(k));
        jj = mod(id-1,ny)+1;
        ii = (id -jj)/ny+1;
%         [s k nodes(k) id ii jj]
        % coordinates for patch
        x1=X(ii);
        x2=X(ii+1);
        y1=Y(ii,jj);
        y2=Y(ii,jj+1);
        vert = [x1 y1; x2 y1; x2 y2; x1 y2];
        face = [1 2 3 4];
        patch('Faces',face,'Vertices',vert,'FaceColor',col{quad(k)},'FaceAlpha',gradation(mod(as(k)-1,2)+1));
%         patch('Faces',face,'Vertices',vert,'FaceColor',col{quad(k)},'FaceAlpha',gradation2(as(k)+1));
    end
    title(sprintf('Stage %d',s));
    if n_stages <= 999
        filename=sprintf('%s%d.png',filename_png,s);
        print('-dpng',filename);
        drawnow
        frame = getframe(101);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if s == 1
            imwrite(imind,cm,filename_gif,'gif', 'Loopcount',inf);
        else
            imwrite(imind,cm,filename_gif,'gif','WriteMode','append');
        end
    end
%     pause
end
