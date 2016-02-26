The run_SPECT_tests.sh scripts uses input data from a SimSET simulation performed at the Univ of Barcelona. It
reconstructs this with various settings and compares the results with reconstructions that were
done by the STIR-SPECT team when developing the code. These are in the org directory.

Sadly, when the files were generated, we made a mistake such that the attenuation map is actually rotated
over 90 degrees and flipped in z-direction (as compared to the reconstructed image). 
This was spotted by Katherine Royston. This means that the "original" reconstructions are actually incorrect. 

However, as the test was not designed to check for correctness, but consistency over different STIR versions,
Kris Thielemans decided to keep the files as they are.

Our apologies for the confusion that this can cause.
