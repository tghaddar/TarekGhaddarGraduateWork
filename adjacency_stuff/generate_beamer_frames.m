clc;

a='\begin{frame}\frametitle{Sweep process examples: 1 angle set}';
b='2D processor layout of $20 \times 20$ for a random partition. All angles (in a quadrant) are done in 1 angle set)';
c='\begin{minipage}[c]{.99\textwidth}';
d='\includegraphics[width=.9\textwidth]{\FigDir/sweeps_png/sweep_worst_20x20_as1_dog_';
e='\end{minipage}';
f='\end{frame}';
g='%';

N=230;
for i=1:N
    fprintf('%s \n',a);
    fprintf('%s \n',b);
    fprintf('%s \n',c);
    fprintf('%s%d%s \n',d,i,'.png}');
    fprintf('%s \n',e);
    fprintf('%s \n',f);
    fprintf('%s \n',g);
end