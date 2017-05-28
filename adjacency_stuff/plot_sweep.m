function plot_sweep(wave,ord,X,Y,nx,ny,quad)

n_stages = length(wave);

figure(100); clf;
hold on;
% superimpose the subset mesh
for i=1:nx+1
    line([X(i),X(i)],[0,1])
    if i<=nx
        for j=1:ny+1
            line([X(i),X(i+1)],[Y(i,j),Y(i,j)])
        end
    end
end

for s=1:n_stages
    nodes = wave{s};
    % id = (i-1)*ny+j
    for k=1:length(nodes)
        id = ord(nodes(k));
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
        patch('Faces',face,'Vertices',vert,'FaceColor','blue','FaceAlpha',0.5);
        % patch([x1 x2 x2 x1],[y1 y1 y2 y2]);
    end
    title(sprintf('Quadrant %d, stage %d',quad,s));
    if n_stages <= 999
        filename=sprintf('./sweep_pix/sweep_stage_%d_%0.3d.png',quad,s);
        print('-dpng',filename);
    end
%     pause
end
