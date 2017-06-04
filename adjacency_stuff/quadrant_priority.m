function [to_do,to_lag,conflict_,stop_sweep] = quadrant_priority(to_do,to_lag,conflict,nx,ny)

% if nx>=ny
%     % further quadrant priority
%     q_prio = [1 2 3 4];
% else
%     q_prio = [1 3 2 4];
% end
% 
stop_sweep=false;

conflicted=length(conflict);

for k=1:conflicted
    
    n=length(conflict{k}(:,1));
    
%     if n~=2
%         conflict{k}
%         warning('only resolving conflicts pair-by-pair for now; in %s',mfilename);
% %         stop_sweep=true;
% %         break
%     end
    
    % quadrant IDs
    %  2---4
    %  |   |
    %  1---3
    q1=conflict{k}(1,2);
    q2=conflict{k}(2,2);
    if q1==q2
        conflict{k}(1,:)
        conflict{k}(2,:)
        error('same node, same quadrant, it cannot be due to as. Issue in %s',mfilename);
    end
    if (q1==1 && q2==4) || (q1==4 && q2==1)
        collision_type='diagonal-';
    elseif (q1==2 && q2==3) || (q1==3 && q2==2)
        collision_type='diagonal+';
    elseif (q1==1 && q2==2) || (q1==2 && q2==1) || (q1==3 && q2==4) || (q1==4 && q2==3)
        collision_type='horizontal';
    elseif (q1==1 && q2==3) || (q1==3 && q2==1) || (q1==2 && q2==4) || (q1==4 && q2==2)
        collision_type='vertical';
    else
        error('unknown collision type in %s',mfilename);
    end
    
    % (i,j) pair for each conflicted node
    id1 = conflict{k}(1,1);
    id2 = conflict{k}(2,1);
    if id1~=id2
        error('nodes should have the same IDs: id1=%d, id2=%d, in %s',id1,id2,mfilename);
    end
    j = mod(id1-1,ny)+1;
    i = (id1 -j)/ny+1;
    
    switch collision_type
        case 'diagonal+'
            % for a given (i,j), we count the # of hops to go to corner 2:
            % (i-1)+(ny-j)
            % likewise, to reach corner 3, we need: (nx-i)+(j-1)
            % the pair (i,j) for which the number of hops is equal is given
            % by: (i-1)+(ny-j) = (nx-i)+(j-1), or j = i + (ny-nx)/2
            % if, for a given (i,j), j-i-(ny-nx)/2>0, we are closer to
            % corner 2, if <0, we are closer to corner 3. if =0, same
            % distance, we need another rule to resolve the conflict
            sgn = j-i-(ny-nx)/2;
            if sgn>0 % do quadrant-2 first
                if q1==2
                    i_to_do =1;
                    i_to_lag=2;
                else
                    i_to_do =2;
                    i_to_lag=1;
                end
            elseif sgn<0 % do quadrant-2 last
                if q1==2
                    i_to_do =2;
                    i_to_lag=1;
                else
                    i_to_do =1;
                    i_to_lag=2;
                end
            else
                % cannot decide yet
                i_to_do =[];
            end
            
        case 'diagonal-'
            % for a given (i,j), we count the # of hops to go to corner 1:
            % (i-1)+(j-1)
            % likewise, to reach corner 4, we need: (nx-i)+(ny-j)
            % the pair (i,j) for which the number of hops is equal is given
            % by: (i-1)+(j-1) = (nx-i)+(ny-j), or j = -i + (ny+nx)/2+1
            % if, for a given (i,j), j+i-(ny+nx)/2-1>0, we are closer to
            % corner 4, if <0, we are closer to corner 1. if =0, same
            % distance, we need another rule to resolve the conflict
            sgn = j+i-(ny+nx)/2-1;
            if sgn>0 % do quadrant-4 first
                if q1==4
                    i_to_do =1;
                    i_to_lag=2;
                else
                    i_to_do =2;
                    i_to_lag=1;
                end
            elseif sgn<0 % do quadrant-4 last
                if q1==4
                    i_to_do =2;
                    i_to_lag=1;
                else
                    i_to_do =1;
                    i_to_lag=2;
                end
            else
                % cannot decide yet
                i_to_do =[];
            end

        case 'horizontal'
            if mod(ny,2)==0 % even ny
                if j<=ny/2 % the bottom quadrants go first
                    if q1==1 || q1==3
                        i_to_do =1;
                        i_to_lag=2;
                    else
                        i_to_do =2;
                        i_to_lag=1;
                    end
                else % the top quadrants go first
                    if q1==2 || q1==4
                        i_to_do =1;
                        i_to_lag=2;
                    else
                        i_to_do =2;
                        i_to_lag=1;
                    end
                end
            else % odd ny
                if j<(ny+1)/2 % the bottom quadrants go first
                    if q1==1 || q1==3
                        i_to_do =1;
                        i_to_lag=2;
                    else
                        i_to_do =2;
                        i_to_lag=1;
                    end
                elseif j>(ny+1)/2 % the top quadrants go first
                    if q1==2 || q1==4
                        i_to_do =1;
                        i_to_lag=2;
                    else
                        i_to_do =2;
                        i_to_lag=1;
                    end
                else
                    % cannot decide yet
                    i_to_do =[];
                end
            end
            
        case 'vertical'
            if mod(nx,2)==0 % even nx
                if i<=nx/2 % the left quadrants go first
                    if q1==1 || q1==2
                        i_to_do =1;
                        i_to_lag=2;
                    else
                        i_to_do =2;
                        i_to_lag=1;
                    end
                else % the right quadrants go first
                    if q1==3 || q1==4
                        i_to_do =1;
                        i_to_lag=2;
                    else
                        i_to_do =2;
                        i_to_lag=1;
                    end
                end
            else % odd nx
                if i<(nx+1)/2 % the left quadrants go first
                    if q1==1 || q1==2
                        i_to_do =1;
                        i_to_lag=2;
                    else
                        i_to_do =2;
                        i_to_lag=1;
                    end
                elseif i>(nx+1)/2 % the right quadrants go first
                    if q1==3 || q1==4
                        i_to_do =1;
                        i_to_lag=2;
                    else
                        i_to_do =2;
                        i_to_lag=1;
                    end
                else
                    % cannot decide yet
                    i_to_do =[];
                end
            end
            
        otherwise
            error('unknown collision type %s in %s',collision_type,mfilename);
            
    end % end switch
    
    % apply
    if ~isempty(i_to_do)
        to_do =[to_do ; conflict{k}(i_to_do ,:)];
        to_lag=[to_lag; conflict{k}(i_to_lag,:)];
        conflict{k}=[];
    end
    
    % move on to the next conflict to be resolved using this rule
end

% completed remove empty cells
kk=0;conflict_=[];
for k=1:conflicted
    if ~isempty(conflict{k})
        kk=kk+1;
        conflict_{kk}=conflict{k};
    end
end

