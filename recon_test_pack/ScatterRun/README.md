# Instructions on executing this test

The files in this folder can be used as a manual test for the scatter estimation process and as a template for running your scatter estimations.

## Running as a test

In order to run it as a test you have to manual download the NEMA file from https://zenodo.org/record/1304454 and extract in `NEMA_IQ`. Then run the command `scatter_estimate ScatterEstimation.par`. 

## Using as a template

The files in the folder have default values which should be alright for most people. Place your projection, multiplicative and additive data in the folder and set the appropriate names in the `ScatterEstimation.par` file. 

## Data preparation

### Creation of projection data

The downloaded data are in listmode format. In order to convert them in projection data the following commands should be executed in the extraction folder: 

```bash
create_projdata_template tmpl_scanner
cp ../par_files/lm_to_projdata.par . 
lm_to_projdata lm_to_projdata.par
```

Now there two new files names as ```nema_proj_f1g1d0b0.hs``` and ```nema_proj_f1g1d0b0.s``` which are the emission projection data.

### Attenuation correction

In the downloaded folder a CT image of the phantom is included. If in the scatter estimation parameter file you set only the name of the image then the attenuation correction factors will be calculated for you automatically. 

Next time you run the estimation you can save some time by setting the recalculation off, as the data will have already been stored.

*As of the time of writing STIR cannot read Siemens image header files correctly. Therefore a header file for the attenuation image is supplied in the `par_files` folder. Just copy and paste the `attenuation.hv` in the `NEMA_IQ` folder.* 

### Randoms 

In this dataset a randoms sinogram is not provided.

## Running the estimation

If you want to run in debug mode, which will generate a lot of output files set the `run in debug mode := 1` in `ScatterEstimation.par` otherwise turn to 0. 
