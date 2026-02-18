The run_PinholeSPECT_tests.sh script uses input data from a GATE simulation performed by Matthew Strugari. 
It reconstructs this with various settings and compares the results with reconstructions that were done by 
the STIR-PinholeSPECT team when developing the code. These are in the org directory.



The simulated system was the Cubresa Spark preclinical SPECT scanner. Simulation was based on a custom 
acrylic NEMA mouse phantom (D = 25.4 mm, L = 60 mm) with three precision capillary tubes (D_inner = 0.4 mm, 
D_outer = 0.8 mm, L = 60 mm) containing ~10 MBq each. The phantom orients the line sources with one at the 
center and two separated by 90 deg with a 10 mm radial offset. In STIR's coordinate system, the simulation 
was configured with the phantom centered along the scanner's axis with one line source placed at 
(x,y) = (0 mm, +10 mm) and the other at (x,y) = (-10 mm, 0 mm). A 91 minute acquisition was simulated with 
91 projections over 270 deg in 3 deg increments, and the detector was rotated in the CCW positive-direction 
starting with the first projection at 180 deg (i.e., below the subject).



Illustration when looking from bed into gantry:


    *   *
        
        *

      _____ ->
