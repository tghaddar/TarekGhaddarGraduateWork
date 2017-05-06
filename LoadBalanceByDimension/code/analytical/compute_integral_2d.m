function F = compute_integral_2d(fh,x,y)

% compute sizes
Ncolx=length(x)-1;
Ncoly=length(y)-1;

% compute the integral
F=zeros(Ncolx,Ncoly);
for i=1:Ncolx
    for j=1:Ncoly
        F(i,j)=integral2(fh,x(i),x(i+1),y(j),y(j+1));
    end
end

