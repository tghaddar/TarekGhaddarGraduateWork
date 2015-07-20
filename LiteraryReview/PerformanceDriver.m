function [RealEff,A_final,RealEff_KBA,A_finalKBA,A_finalVolumetric,RealEff_Volumetric] = PerformanceDriver(M_L,n)
%A code that optimizes the performance model for each processor layout
%INPUT:
%M_L = A latency factor that dictates whether the model is high or low
%overhead.
%n: This is simply the n in Sn. We can calculate the number of angles per
%octant from it.
%P: A 15x3 matrix that stores the breakdown of processors in x, y,
%          and z for each total processor count.
P = [1,1,1;2,2,2;8,4,2;16,16,2;32,16,2;32,32,2;64,32,2;64,64,2;128,64,2;128,128,2;256,128,2;256,256,2;512,256,2;512,512,2;1024,512,2];
%P_KBA: A 15x3 matrix that stores the breakdown of processors for KBA
%partitioning.
P_KBA = [1,1,1;4,2,1;8,8,1;32,16,1;32,32,1;64,32,1;64,64,1;128,64,1;128,128,1;256,128,1;256,256,1;512,256,1;512,512,1;1024,512,1;1024,1024,1];
%P_Volumetric
P_Volumetric = [1,1,1;2,2,2;4,4,4;8,8,8;10,10,10;12,12,12;16,16,16;20,20,20;26,26,26;32,32,32;40,40,40;50,50,50;64,64,64;80,80,80;100,100,100];
%N: A 15x3 matrix that stores the breakdown of cells in x,y, and z.
N = [16,16,16;32,32,32;64,64,64;128,128,128;128,128,256;256,128,256;256,256,256;256,256,512;512,256,512;512,512,512;512,512,1024;1024,512,1024;1024,1024,1024;1024,1024,2048;2048,1024,2048];
%The processor array, stores the breakdown of processors in x, y, and z.
%Storing the efficiency for each total number of processors
RealEff = zeros(size(P,1),1);
RealEff_KBA = zeros(size(P_KBA,1),1);
%The time it takes to communicate a double (ns).
T_byte = 4.47;
%The serial grind time (ns).
T_grind = 3292;
%The latency time (ns)
T_latency = 4114.74;
%Looping over processor counts to optimize
%N_bytes = [0 1280 960 640 480 480 480 320 320 320 160 160 160 80 80];
%The parallel efficiency for a serial case is obviously 1.
RealEff(1) = 1.0;
RealEff_KBA(1) = 1.0;
P_total(1) = 1;
%Stores the final aggregation
A_final = zeros(size(P,1),3);
A_finalKBA = zeros(size(P_KBA,1),3);
A_final(1,:) = [1 1 1];
A_finalKBA(1,:) = [1 1 1];
%The number of angles per octant.
M = n*(n+2)/8;
%For the purposes of the model, we need M to be an integer. This limits us
%to S2, S4, S6, S8, etc..
if (mod(n*(n+2),8) ~= 0)
    error('M must be an integer for this performance model to work, change the value of n to an even number');
end
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
                A = [a b c M 1];
                N_Current = N(i,:);
                P_Current = P(i,:);
                N_bytes = 8*4*(A(1)+A(2))*A(3)*A(4)*A(5);
                eff = PerformanceModel(M_L,T_latency,T_byte,N_bytes,T_grind,P_Current,A,N_Current,M);
                if (eff > opt_eff)
                    opt_eff = eff;
                    RealEff(i) = opt_eff;
                    A_final(i,:) = [a b c];
                end
            end
        end
    end
end

%For KBA
for i = 2:size(P,1)
    P_total(i) = P_KBA(i,1)*P_KBA(i,2)*P_KBA(i,3);
    %This is the iterative processess to determine the optimized
    %aggregation factors for the current cell and processor layouts.
    %In each direction, the start point is A_z = 1, and the end point is
    %A_z = N_z/P_z.
    %For KBA, A_x and A_y are usually set.
    A_x = N(i,1)/P_KBA(i,1);
    A_y = N(i,2)/P_KBA(i,2);
    Az_start = 1; Az_end = N(i,3)/P_KBA(i,3);
    %A dummy starting optimal efficiency
    opt_eff = 0;
    for c = Az_start:Az_end
        %KBA takes A_m = A_g = 1.
        A = [A_x A_y c 1 1];
        N_Current = N(i,:);
        P_Current = P_KBA(i,:);
        N_bytes = 8*4*(A(1)+A(2))*A(3)*A(4)*A(5);
        eff = PerformanceModel(M_L,T_latency,T_byte,N_bytes,T_grind,P_Current,A,N_Current,M);
        if (eff > opt_eff)
            opt_eff = eff;
            RealEff_KBA(i) = opt_eff;
            A_finalKBA(i,:) = [a b c];
        end
    end
end

%For Volumetric
P_totalVolumetric = zeros(size(P,1),1);
P_totalVolumetric(1) = 1;
RealEff_Volumetric(1) = 1.0;
A_finalVolumetric(1,:) = [1 1 1];
for i = 2:size(P,1)
    P_totalVolumetric(i) = P_Volumetric(i,1)*P_Volumetric(i,2)*P_Volumetric(i,3);
    %Volumetric takes all A_u = N_u/P_u.
    A_x = N(i,1)/P_Volumetric(i,1);
    A_y = N(i,2)/P_Volumetric(i,2);
    A_z = N(i,3)/P_Volumetric(i,3);
    A = [A_x A_y A_z M 1];
    N_Current = N(i,:);
    P_Current = P_Volumetric(i,:);
    N_bytes = 8*4*(A(1)+A(2))*A(3)*A(4)*A(5);
    eff = PerformanceModel(M_L,T_latency,T_byte,N_bytes,T_grind,P_Current,A,N_Current,M);
        RealEff_Volumetric(i) = opt_eff;
        A_finalVolumetric(i,:) = [a b c];
end

semilogx(P_total,RealEff,'+-k',P_total,RealEff_KBA,'o--r',P_totalVolumetric,RealEff_Volumetric,'x:b');
title('Parallel Efficiency of PDT''s Performance Model');
xlabel('Processors');
ylabel('Parallel Efficiency');
ylim([0,1]);

end

