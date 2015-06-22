function [eff,P_total] = PerformanceModel(M_L,T_latency,T_byte,N_bytes,T_grind,A,M,G)
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

%The time to communicate between processors
T_comm = M_L*T_latency + T_byte*N_bytes;
%The product of all aggregation factors multiplied together
aggregation_product = 1;
for i = 1:length(A)
   aggregation_product = aggregation_product*A(i); 
end
%The time to communicate a task
T_task = aggregation_product*T_grind;
%Filling out the processor arrays
P_z = 2;
P = zeros(15,2);
P(:,2) = P_z;
P_total = zeros(15,1);
P_total(1) = 8;
P(1,1) = P_total(1) - P_z;
for i = 2:15
    P_total(i) = P_total(i-1)*2;
    P(i,1) = P_total(i) - 2;
end

%The aggregation in Z
A_z = A(3);
%The number of cellsets
N_z = P_z*A_z*4;
N_k = N_z/(P_z*A_z);
%The parallel efficiency
eff = zeros(15,1);
for i = 1:15
    eff(i) = ((1 + (P(i,1)-4 + N_k*(P_z-2))/((8*M*G*N_k)/(A(4)*A(5))))*(1+T_comm/T_task)) ^(-1);
end

plot(P_total,eff);


end

