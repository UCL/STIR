
# Reconstructing data from GE SIGNA PET/MR Scanner

Author: Palak Wadwha
Edits: Kris Thielemans

SPDX-License-Identifier: Apache-2.0
See STIR/LICENSE.txt for details

WARNING: In this release, the reconstructed images are x,z flipped, rotated and
shifted w.r.t. to the scanner DICOMS.

These instructions use a  template that was created using the `create_projdata_template` utility using the following inputs:
```
Enter the name of the scanner : GE Signa PET/MR
Mashing factor for views : 1
Is the data arc-corrected?: N
Number of the tangential positions: 357
Span value: 2
Max. ring difference acquired: 44
```

1. Unlisting GE HDF5 listmode file: In order to unlist the uncompressed list mode file from GE
SIGNA PET/MR scanner, use `lm_to_projdata` with `lm_to_projdata.par` file.
An example of the parameter file is supplied.

2. If the extracted HDF5 file from the scanner is an uncompressed sinogram instead of the list
mode, we recommend to convert it first to Interfile using:
```
stir_math -s output.hs RDF_filename
```
where `RDF_filename` is the uncompressed RDF sinogram file extracted from the scanner.

3. Randoms Correction: For the randoms correction sinogram, utility
`construct_randoms_from_GEsingles` can be used as :
```
construct_randoms_from_GEsingles out_filename listmode_filename template.hs
```
where the “listmode_filename” is the list mode file extracted from the scanner.

4. Normalisation Correction:
For the normalisation correction sinogram, the utility `correct_projdata` with
an example `correct_projdata_for_norm.par` file
is used to construct normalisation sinogram that
corrects for the crystal efficiency factors and geometric factors.
```
correct_projdata correct_projdata_for_norm.par
```
This uses the `*norm.rdf` (note: `*geo.rdf` is not necessary). In the current
release, we do not yet use the "well-counter calibration files".
Therefore, the reconstructed image will only be proportional to the
console images.

5. Attenuation Correction:
You will need to convert the MRAC (including hardware
such as bed)  obtained from the scanner. Instructions for this are currently
out-of-scope. This image needs to be converted to mu-values (in units cm^-1).
You can try the `nm_signa2mu` program of the `pet-rd-tools` for this.
Unfortunately, in this release, the obtained mu-map needs to be x,z flipped and 
rotated by 5.23 degrees in (anti-clockwise) direction. We recommend
reconstructing the data with attenuation correction, and using
`register_GEAC.sh`.


These steps are made somewhat easier by the provided example scripts.