function [true_error] = CalculateFluxError()
%Calculates the flux info.

[cell_intersections,cell_verts,cell_info] = ProcessCellInfo();

num_cells = size(cell_info,1);

error_sum = 0;

for i = 1:num_cells
    if (isempty(cell_intersections{i}))
       continue 
    else
        verts_and_fluxes = cell_verts{i};
        intersections = cell_intersections{i};
        num_verts = cell_info(i,1);
        
        x_sort = zeros(2,1);
        y_sort = zeros(2,1);
        
        x_begin = intersections(1,1);
        y_begin = intersections(1,2);
        x_end = intersections(2,1);
        y_end = intersections(2,2);
        face_begin = intersections(1,3);
        face_end = intersections(2,3);
        
        
        if (x_begin > x_end);
           xsort1 = x_end;
           xsort2 = x_begin;
           x_begin = xsort1;
           x_end = xsort2;
           
           ysort1 = y_end;
           ysort2 = y_begin;
           y_begin = ysort1;
           y_end = ysort2;
           
           facesort1 = face_end;
           facesort2 = face_begin;
           face_begin = facesort1;
           face_end = facesort2;
           
        end
        
        
        %What we need to calculate.
        flux_begin = 0;
        flux_end = 0;
        
        if (num_verts == 3)%Just solving on a triangle.
            %We have to check which face the endpoints exist on in order to
            %calculate the fluxes at x/y_begin/end
            
            
            
            x_coords = verts_and_fluxes(:,1);
            y_coords = verts_and_fluxes(:,2);
            fluxes = verts_and_fluxes(:,3);  
            
            
            %Calculating flux_begin.
            x1 = x_coords(face_begin);
            y1 = y_coords(face_begin);
            flux1 = fluxes(face_begin);
            
            if (face_begin == num_verts)
                x2 = x_coords(1);
                y2 = y_coords(1);
                flux2 = fluxes(1);
            else
                x2 = x_coords(face_begin+1);
                y2 = y_coords(face_begin+1);
                flux2 = fluxes(face_begin+1);
            end
            
            face_length = sqrt( (x2-x1)^2 + (y2 - y1)^2  );
            distance1 = sqrt( (x2-x_begin)^2 + (y2 - y_begin)^2   );
            distance2 = sqrt( (x1-x_begin)^2 + (y1 - y_begin)^2   );
            
            flux_begin = (flux1*distance1 + flux2*distance2)/face_length;
            
            %Calculating flux_end
            %Calculating flux_begin.
            x1 = x_coords(face_end);
            y1 = y_coords(face_end);
            flux1 = fluxes(face_end);
            
            if (face_end == num_verts)
                x2 = x_coords(1);
                y2 = y_coords(1);
                flux2 = fluxes(1);
            else
                x2 = x_coords(face_end+1);
                y2 = y_coords(face_end+1);
                flux2 = fluxes(face_end+1);
            end
            
            face_length = sqrt( (x2-x1)^2 + (y2 - y1)^2  );
            distance1 = sqrt( (x2-x_end)^2 + (y2 - y_end)^2   );
            distance2 = sqrt( (x1-x_end)^2 + (y1 - y_end)^2   );
            
            flux_end = (flux1*distance1 + flux2*distance2)/face_length;
            
            %Getting a quadrature to evaluate an integral.
            [nodes,weights] = lgwt(2,x_begin,x_end);
            %x_begin,x_end
            nodes = sort(nodes);
            for j = 1:length(nodes)
                xq = nodes(j);
                wq = weights(j);
                s = 3.5*expint(2,sym(5*xq));
                flux = vpa(s,10);
                %flux
                error_sum = error_sum + wq*((flux_begin*(x_end - xq) + flux_end*(xq - x_begin))/(x_end-x_begin) - flux )^2;
                %error_sum
            end
            
            
        else %The number of vertices is greater than 3 and we have to do some additional things.
            
        end
        
    end
end

true_error = sqrt(error_sum);


end

