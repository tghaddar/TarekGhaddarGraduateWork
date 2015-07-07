function [] = PerformanceDriver(N,P)
%A code that optimizes the performance model for each processor layout
%INPUT: 
%P: A 15x3 matrix that stores the breakdown of processors in x, y,
%          and z for each total processor count.
%N: A 15x3 matrix that stores the breakdown of cells in x,y, and z.

%Storing the efficiency for each total number of processors
RealEff = zeros(size(P,1),1);
%The time it takes to communicate a double (ns).
T_byte = 4.47;
%Looping over processor counts to optimize 
for i = 1:size(P,1)
   %Total number of processors
   P_total = P(i,1)*P(i,2)*P(i,3);
   %The total number of cells at 4096 cells per core
   
end

end

