function plot_sweep(wave,ord,X,Y,nx,ny)

n_stages = length(wave);

figure(100); clf;

for s=1:n_stages
    nodes = wave{s};
    % id = (i-1)*ny+j
    for k=1:length(nodes)
        id = ord(nodes(k));
        ii = floor(id/ny)+1;
        jj = id -(ii-1)*ny;
        % coordinates for patch
        x1=X(ii);
        x2=X(ii+1);
        y1=Y(ii,jj);
        y2=Y(ii,jj+1);
        patch([x1 x2 x2 x1],[y1 y1 y2 y2]);
    end
    title(sprintf('stage %d',s));
    filename=sprintf('./sweep_pix/sweep_stage_%0.3d.png',s);
    print('-dpng',filename);
end
