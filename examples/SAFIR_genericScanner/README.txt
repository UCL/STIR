The PET scanner prototype of the SAFIR project does not consists of symmetric 
detector blocks. This led to the need for a new scanner geometry in STIR. It 
is implemented named generic geometry. In short it uses the coordinates 
provided by an external text file instead of the ones computed internally by 
STIR itself (based on the symmetry of the detector).

The files in this directory are to test the new implementation, to understand 
the changes and as model for later usage. 

The bash script 'test.sh' runs the methods 'lm_to_projdata', 'backproject', 
'forward_project' and 'OSMAPOSL' to test the functionality with the new scanner 
geometry. It shows as well how the different parameter files are used. In the 
next sections the different files and methods are explained more precisely.

The first method call 'lm_to_projdata lm_to_projdata.par' turns the listmode 
data in the file 'coincidencesLM.clm.safir' to projdata (a.k.a. sinogram data). 
The parameter file contains the input SAFIR parameter file 
'listmode_input_SAFIR.par' which specifies the above mentioned data file, the 
crystal map file and the projdata file template 'muppet.hs'.
The 'lm_to_projdata.par' parameter file contains as well the projdata file 
template and the output file name. More infos about the other entries can be 
found in the online documentation of STIR and another example file can be found 
in 'example/samples/lm_to_projdata.par'.

The projdata template file can be created with the utility 
'create_projdata_template'. The new entries for this implementation are 'Name of 
crystal map' which defines the name of the file which contains all coordinates 
of the detectors like the example file 'DualRingPrototype_crystal_map.txt' and 
it contains as well the new 'Scanner geometry' option 'Generic'. The 'crystal map'
file should be in the same folder as the template file. For the other entries 
please check the online documentation.

'forward_project' and 'backproject' are the utilities that convert the sinogram 
 data into an image and vice versa.
 
The OSMAPOSL method is an iterative algorithm to get an image from the 
sinogram data. It uses a parameter file as defined in the STIR user guide or 
online documentation. There is as well another example file in 
'example/samples'.
