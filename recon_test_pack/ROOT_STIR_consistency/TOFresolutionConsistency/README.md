# Using TOF Gate data - setting up the single detector time resolution
## Author: Elise Emond

In Gate, you can only define a "single detector time resolution" =/= coincidence time resolution.

If you know the expected coincidence time resolution for a given scanner (e.g. 550 ps for the Discovery 710) you can follow this to calibrate accordingly your Gate scanner.

Command for Gate single detector time resolution: /gate/digitizer/Singles/timeResolution/setTimeResolution in digitiser.

**What you can do:** 

1. Run a Gate simulation with perfect time resolution (in the digitiser, setTimeResolution=0 ns or just comment out that line - default value), use the ascii output in main_D690_centre.mac (it will be read in Python)
2. In getTimingResolution.py, run the last cell (starting from last #%%) - don't forget to modify the output name (it is written Output_centre2Coincidences.dat). You can get the theoretical "detector response" (some kind of triangular response). 
3. Then you can deconvolve the theoretical distribution (Gaussian distribution with expected FWHM, for Discovery 710 550ps) with this detector response to get an estimate of the single detector time resolution (you will obtain a Gaussian distribution after the deconvolution with a slightly lower FWHM: use setTimeResolution = this new FWHM/sqrt(2))
4. Check that this is correct: modify once again the setTimeResolution in the digitiser with the value you just calculated and then run the second to last Python cell in getTimingResolution.py to obtain the coincidence FWHM.

*NB: sadly I had written a nice Mathematica script for convolution/deconvolution but lost it when my computer crashed at the MIC 2017 and forgot to recover this file... You can however use the file in eliseemond/Mathematica/ConvolutionEffectiveEnergy.nb as a model to convolve distributions in Mathematica.*
