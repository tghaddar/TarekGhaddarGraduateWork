function [RealEff] = PerformanceDriver(M_L)
%A code that optimizes the performance model for each processor layout
%INPUT:
%M_L = A latency factor that dictates whether the model is high or low
%overhead.
%P: A 15x3 matrix that stores the breakdown of processors in x, y,
%          and z for each total processor count.
P = [1,1,1;2,2,2;8,4,2;16,16,2;32,16,2;32,32,2;64,32,2;64,64,2;128,64,2;128,128,2;256,128,2;256,256,2;512,256,2;512,512,2;1024,512,2];
%N: A 15x3 matrix that stores the breakdown of cells in x,y, and z.
N = [16,16,16;32,32,32;64,64,64;128,128,128;128,128,256;256,128,256;256,256,256;256,256,512;512,256,512;512,512,512;512,512,1024;1024,512,1024;1024,1024,1024;1024,1024,2048;2048,1024,2048];
%The processor array, stores the breakdown of processors in x, y, and z.
%Storing the efficiency for each total number of processors
RealEff = zeros(size(P,1),1);
%The time it takes to communicate a double (ns).
T_byte = 4.47;
%The serial grind time (ns).
T_grind = 3292;
%The latency time (ns)
T_latency = 4114.74;
%Looping over processor counts to optimize
N_bytes = [0 1280 960 640 480 480 480 320 320 320 160 160 160 80 80];
%The parallel efficiency for a serial case is obviously 1.
RealEff(1) = 1.0;
P_total(1) = 1;
for i = 2:size(P,1)
    P_total(i) = P(i,1)*P(i,2)*P(i,3);
    %This is the iterative processess to determine the optimized
    %aggregation factors for the current cell and processor layouts.
    %In each direction, the start point is A_u = 1, and the end point is
    %A_u = N_u/P_u for u = x,y, or z.
    Ax_start = 1; Ax_end = N(i,1)/P(i,1);
    Ay_start = 1; Ay_end = N(i,2)/P(i,2);
    Az_start = 1; Az_end = N(i,3)/P(i,3);
    %A dummy starting optimal efficiency
    opt_eff = 0;
    for a = Ax_start:Ax_end
        for b = Ay_start:Ay_end
            for c = Az_start:Az_end
                A = [a b c 10 1];
                N_Current = N(i,:);
                P_Current = P(i,:);
                eff = PerformanceModel(M_L,T_latency,T_byte,N_bytes(i),T_grind,P_Current,A,N_Current);
                if (eff >= opt_eff)
                    opt_eff = eff;
                    RealEff(i) = opt_eff;
                end
            end
        end
    end
end

semilogx(P_total,RealEff,'+-k');
title('Parallel Efficiency of PDT''s Performance Model');
xlabel('Processors');
ylabel('Parallel Efficiency');
ylim([0,1]);

end

