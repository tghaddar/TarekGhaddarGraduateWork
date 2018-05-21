function [cell_intersections,cell_verts,cell_info] = ProcessCellInfo()
%Processes PDT output.

%Stores x,y,and flux value at the vertex
vertex_info = csvread('vertex_info.csv');
%Stores number of vertices, x center, and y center
cell_info = csvread('cell_info.csv');

%The number of cells in question.
num_cells = size(cell_info,1);

%A cell array that will store intersection information for each cell.
cell_intersections = cell(num_cells,1);
%A cell array that stores the x_coordinates, y_coordinates, and fluxes for
%each cell
cell_verts = cell(num_cells,1);

%Looping over all cells.
vert_counter = 1;
for i = 1:num_cells
  %The number of vertices in this cell
  num_verts = cell_info(i,1);
  
  %Pulling the x and y coordinates of the cell.
  x_coords = zeros(num_verts,1);
  y_coords = x_coords;
  fluxes = x_coords;
  for j = 1:num_verts
      x_coords(j) = vertex_info(vert_counter,1);
      y_coords(j) = vertex_info(vert_counter,2);
      fluxes(j) = vertex_info(vert_counter,3);
      total_info = [x_coords y_coords fluxes];
      cell_verts{i} = total_info;
      vert_counter = vert_counter + 1;
  end

  %We have to get the x_min and x_max of this cell to calculate the intersection of this polygon with y = 0.45 to get the appropriate quadrature points.
  num_faces = num_verts;
  %The minimum and maximum x coordinates
  x_min = min(x_coords);
  x_max = max(x_coords);

  %The two intersection points with the cell, and which face they
  %intersect.
  x_int = zeros(2,1);
  y_int = x_int;
  counter = 0;
  intersections = zeros(2,3);
  x1 = 0;
  x2 = 0; 
  y1 = 0; 
  y2 = 0;
  for a = 1:num_faces
    if (a < num_faces)
      x1 = x_coords(a); y1 = y_coords(a);
      x2 = x_coords(a+1); y2 = y_coords(a+1);
    else
      x1 = x_coords(a); y1 = y_coords(a);
      x2 = x_coords(1); y2 = y_coords(1);
    end
    %x1,x2,y1,y2

    %Slope of this line.
    m = (y2 - y1)/(x2 - x1);
    if (x2 == x1)
      x_int(counter+1) = x1;
      y_int(counter+1) = 0.45;
      intersections(counter+1,3) = a;
      counter = counter + 1;
    else
      x3 = (0.45 - y1)/m + x1;
      if (x3 <= x_max && x3 >= x_min)
          x_int(counter+1) = x3;
          y_int(counter+1) = 0.45;
          intersections(counter+1,3) = a;
          counter = counter + 1;
      end
    end
    %x_int,y_int

    %If the slope is zero, we won't have a viable x intersection
%     if (m == 0)
%       continue
%     end    

    

    %If we've already found both intersection points, terminate looping over the faces of the cell
    if (counter == 2)
      intersections(1,1:2) = [x_int(1) y_int(1)];
      intersections(2,1:2) = [x_int(2) y_int(2)];
      cell_intersections{i} = intersections;
      break
    end
  end %End looping over cell faces.

end

end

