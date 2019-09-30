## TOF consistency checks for STIR
# Author: Elise Emond

Files in the folder were created to check:
* Whether the TOF STIR implementation was correct. To do so you need:
 1. To run `./run_pretest_script.sh` in the terminal to create the Gate root files for different point sources.
 2. Run the STIR test: `src/recon_test/test_consistency_root`. This test should tell you whether it failed or not, using centres of mass corresponding to the maximum value for a detection bin (defined by LOR + time bin). If failed, the TOF backprojection is incorrect.
 3. The image coordinates corresponding to the centres of mass were written as a txt file by the previous test. You can plot using `make_plot.py`.
* To check if the time resolution defined in the Gate digitiser for a single detector corresponds to the time resolution in STIR scanner template. Have a look at `TOFresolutionConsistency/README.md` for more information.
