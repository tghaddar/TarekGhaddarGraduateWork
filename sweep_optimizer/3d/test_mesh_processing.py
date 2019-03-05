from mesh_processor import analytical_mesh_integration

#A test function.
f = lambda z,y,x: pow(x,2) + pow(y,2) + pow(z,2)

result = analytical_mesh_integration(f,0,1,0,1,0,1)

