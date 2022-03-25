# TOF consistency checks for STIR
## Authors: Elise Emond & Robert Twyman

Files in the folder were created to check:
* Whether the TOF STIR implementation was correct. This is also used to test non-TOF ROOT and STIR consistency. 

Methodology
 1. Generate the ROOT data. 
     1. Run `./run_pretest_script.sh` in the terminal to generate the ROOT files (requires Gate) for different point sources, or
     2. Download the ROOT data and proceed without Gate simulation.
     
 2. Run the STIR test: `src/recon_test/test_view_offset_root`.
    This test should tell you whether it failed or not by testing if the LOR passes by, 
    or close to, the original point source position.
