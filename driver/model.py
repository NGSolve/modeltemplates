model = {
    "type" : "static",
    "mesh" : "cube.vol",
    "boundarydisplacement" : { "left" : (0,0,0) },
    "boundaryforce" : { "right" : (0,0,1) },
    "nonlinear" : False,
    "order" : 3
    }

# boundarydisplacement = { "left" : (0,0), "right" : loadfactor*(0,1) },
            
