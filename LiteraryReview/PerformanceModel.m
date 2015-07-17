function [eff] = PerformanceModel(M_L,T_latency,T_byte,N_bytes,T_grind,P,A,N)
%PERFORMANCEMODEL: Function accepts 6 parameters
%M_L: Latency factor, a function of increased and decreased latency
%T_latency: The message latency times
%T_byte: The additional time to send one byte of message
%N_byte: total number of bytes processor must communicate to downstream
%        neighbors
%T_grind: The time it takes to compute a single cell,direction, and energy
%group
%A: Contains all aggregation factors in a vector stored in the following
%   order: A_x,A_y,A_z,A_m,A_g
%N: The number of cells in each direction in a vector stored in the
%   following order N_x, N_y, N_z
%P is the number of processors 

%The time to communicate between processors
T_comm = M_L*T_latency + T_byte*N_bytes;
%The product of all aggregation factors multiplied together
aggregation_product = 1;
for i = 1:length(A)
   aggregation_product = aggregation_product*A(i); 
end
%The time to communicate a task
T_task = aggregation_product*T_grind;

%The aggregation in Z
A_z = A(3);
%The number of cellsets
N_z = P_z*A_z*4;
N_k = N_z/(P_z*A_z);
%The parallel efficiency
eff = ;


end

