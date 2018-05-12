close all;

x = [0 1 2 3 4 5];
y = [0 4 10 14 20 30];

figure;
plot(x,y,'-o');
ax = gca;
ax.XTick = [0 1 2 3 4 5];
xlabel('x (cm)');
ylabel('Unnormalized CDF of triangles per column');


y1 = [6 6 6 6 6 6];
y2 = [12 12 12 12 12 12];
y3 = [18 18 18 18 18 18];
y4 = [24 24 24 24 24 24];


figure;
hold on;
plot(x,y,'-o');
plot(x,y1,'--',x,y2,'--',x,y3,'--',x,y4,'--');
hold off;
ax = gca;
ax.XTick = [0 1 2 3 4 5];
xlabel('x (cm)');
ylabel('Unnormalized CDF of triangles per column');
