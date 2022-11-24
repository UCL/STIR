# MANUAL for wm\_SPECT\_mph.2.0

Author: Carles Falcon

wm\_SPECT.cpp is a code for creating the weight matrix for multiple
pinhole collimator in SPECT using STIR or other reconstruction packages.
It is written in C compatible with C++ and suitable for STIR
reconstruction.

## Code specifications

The code calculates the contribution of each voxel of the image to each
detector element (one per angle and ring) and includes the geometric
projection (shadow of the collimator hole), the correction for intrinsic
PSF, the correction for depth of the impact, the attenuation inside the
crystal and the correction for the attenuation, full (different for each
bin of PSF) and simple (the whole PSF affected by the same attenuation
factor). The input parameters can be entered either by command line or
through a file. The input parameter file should have a determinate
structure. The parameters are read sequentially whatever is the label
they have. Labels in template file are just information for users.
Information about detector and characteristics are be taken from a text
file that should have a fixed structure. The output is a file containing
a system matrix in STIR or FC format and a text file (matrix header)
with information about the input parameters. The system of reference and
sign criteria are shown at the end of this file.

## Input parameter file

A template of this file is distributed with the code. The following
rules should be taken into account:

- The reading routine do no read the number of argv\[x\]. This value
(or any other comment at the begin of the line) is a label. Parameters
are read and assigned sequentially

- Hash (\#, number sign) ends the previous-to-parameter comment. If no
comment, a single hash symbol is needed to indicate the line is not a
text line but a parameter one. Lines without hash will be skipped when
reading parameters.

- Percent (%) indicates comment. The text to the end of the line will
not be considered in the reading routine

- Hash and percent symbols are designated as comment delimiters of this
file in the header file. They can be modified.

- Double vertical bar indicates encoded options.

- Double slash into braces to separate encoded options. Encoded options
are case sensitive.

- Single slash indicates alternatives for parameters depending on
previous parameters.

- The file should contain all the parameters although some of them can
be not necessaries for that particular matrix. If one line is removed
the index numbers will not match. For instance, you should introduce
something in the attenuation map gap (i.e "no") although not correcting
for attenuation

### Program parameters

**argv\[1\]** is the weight matrix filename.

**argv\[2\]** is the number of columns (int) of the reconstructed
volume. Can be even or odd. x-axis discretization range

**argv\[3\]** is the number of rows (int) of the reconstructed volume.
Can be even or odd.

y-axis discretization range

**argv\[4\]** is the number of slices (int) of the reconstructed volume.
Can be even or odd.

z-axis discretization range. The number of slices of the reconstructed
volume should be the same as the number of slices of the projections

**argv\[5\]** is the in-plane voxel size. Only square voxels are
possible (float cm)

**argv\[6\]** is the volume slice thickness. It should be the same than
the projection slice thickness (float cm)

**argv\[7\]** is the first slice to reconstruct (int). The FOV can be
adjusted in z. First slice is 1, no 0.

**argv\[8\]** is the last slice to reconstruct (int)

**argv\[9\]** is the radius of the object (cm) (float). All the voxels
not belonging to the cylinder defined by this radius are masked

**argv\[10\]** name of the file containing the detector information. See
bellow to know about the structure of this file.

**argv\[11\]** name of the file containing the collimator information.
See bellow to know about the structure of this file.

**argv\[12\]** is the minimum weight to take into account (float,
typically 0.005-0.02). It makes reference just to the geometric (PSF)
part of the weight. After applying the attenuation factor, weight could
be lower than this value

**argv\[13\]** is the number of sigma (float) in case of correction for
intrinsic PSF (typically 1.5 – 2.5). To increase unnecessarily the
number of sigma would produce an increase of the weight matrix size that
would not result in a better PSF correction. PSF are modelled by
Gaussian functions whose extension is infinite. It does not make sense
to take into account very small contribution. A balance between the
precision of the correction for PSF and the size of the matrix and the
time of the reconstruction process should be done.

**argv\[14\]** is the spatial high resolution in which to sample
distributions. The geometric contribution to one bin is the plane
integral of the shadow of the hole within the bin. To easily compute
thousands of integrals, the code pre-calculates the cumulative sum of
the shadow of the hole at high resolution. This parameter indicates the
discretization interval for such functions (float, cm, typically
0.001-0.0001, depending on the voxel size). The smaller this parameter
the longer takes to calculate the correction for depth of impact (linear
proportion)

**argv\[15\]** is the subsampling factor to compute convolutions. If
some cases the Gaussian that describes the intrinsic PSF has just few
voxels. To convolve the shadow of the hole with a low resolution
intrinsic PSF could have low accuracy. The subsampling factor (integer
1-8) reduces temporally the resolution of the PSF to perform more
accurate calculus and then down sample the final PSF to the bin size. It
has a great influence on the computation time.

**argv\[16\]** correction for intrinsic PSF. Possible values are { yes
// no }

**argv\[17\]** correction for depth of the impact PSF. Possible values
are { yes // no }

**argv\[18\]** correction for attenuation. The possible values are {
simple // full // no }. Attenuation is calculated as the negative
exponential of the sum of the length of the projection ray in each
crossed voxel by its attenuation coefficient. It requires an attenuation
map with the same geometric characteristics than the image to be
reconstructed (number of columns, row, slices and voxel dimensions). The
attenuation map should be composed by the attenuation coefficient at
each voxel (float cm-1). The simple option for correction for
attenuation is to consider that the whole PSF suffers the same
attenuation (the attenuation of the central ray). One single factor is
applied to weight the contribution from one voxel to all the bins in a
detection plane. It is a very good approximation for uniform attenuation
maps. The full option means that a different attenuation coefficient is
calculated for each voxel-bin contribution (that obtained along the
voxel-bin pathway). It could be useful for very inhomogeneous
attenuation maps.

**argv\[19\]** is the attenuation map file name. In case of no
correction for attenuation set it to “no”. Values in attenuation map
should be in cm-1.

**argv\[20\]** is a coded parameter to indicate if any mask should be
applied to the volume to reduce the matrix size by removing weights from
voxels that do no contribute to the projections. The cylindrical mask is
done by default. This parameter asks for extra masking. Possible values
are: {att // file // no}:

  - att: attenuation map is used as a mask. No weight is calculated
    where attenuation map is zero (no attenuation=no activity). If the
    attenuation map is obtained from a TAC, very small values of
    attenuation could be set around the patient. To threshold the
    attenuation map could be considered to adjust the mask to the
    patient. NaNs values are set to zero.

  - file: a mask is defined in a file. The mask should have the same
    geometrical characteristics than the image. It could be useful, for
    instance, to remove the weight of voxels from the table (no activity
    but attenuation) or to reduce the matrix size when no attenuation
    for correction is considered. If not, attenuation map should be
    enough

**argv\[21\]** is the name of explicit mask in case argv\[25\] = “file”.
Otherwise, set to “no”

**argv\[22\]** is the coded parameter to define the output matrix
format. Options are { STIR // FC }, for STIR compatible matrix and for
Fruitcake (home made software for tomographic reconstruction not yet
available). The format of STIR matrices is explained in the STIR
tutorial. The format of FC matrices is explained in the appendix 1.

## Detector file

This file contains information about projections. The structure is the
following:

```
Any comment here or anywhere in lines that not contain parameters. Avoid
using two points character since it is reserved to indicate the
following value must be read as a parameter

number of rings: 1

rad(cm): 20

FOVh(cm): 5

FOVv(cm): 5

Nbins: 101

Nslices: 101

#intrinsic PSF#

Sigma(cm): 0.10

Crystal thickness (cm): 0.8

Crystal attenuation coefficient (cm -1): 2.22

\#……repeat for each ring …………\#

Nangles: 8

ang0(deg): 0.

incr(deg): 22.5

z0(cm): 0.

\#…………until here………………\#
```
The labels before ':' is just information for users. The parameters
should keep always the same order because they are read sequentially.
':' indicates the following value is a parameter to read, so it should
be avoided in comments. To take into account:

- There can be several rings of detectors. Each one can have different
configuration

- The radius refers to the front of the crystal (external face). In
case of no correction for depth of impact, the program automatically add
half of the crystal thickness to this value.

-The bin size and slice thickness are obtained dividing the FOV by the
matrix dimensions. FOVh, FOVv (horizontal and vertical) can be
different.

- The code works internally in radiants although the angular parameters
in input files are supposed to be in degrees.

- This file should contain all the parameters although not used for
that particular matrix. If not indices will not match.

- z0 refers to the centre of the volume. See figure at the end of this
file to know about angle criteria.

## Collimator file

This file contains information about projections. The structure is the
following:

```
Any comment here or anywhere in lines that not contain parameters. Avoid
using two points character since it is reserved to indicate the
following value must be read as a parameter

Model (cyl/pol): pol

Collimator radius(cm): 15.

Wall thickness (cm): 1

\#holes\#

Number of holes: 8

nh / ind detel (1-\>Ndet) / x(cm) / y(cm) / z(cm) / shape (rect-round) /
size1(cm) / size2(cm) / angx (deg) /angz(deg) / accx(deg) / accz(deg)

h1: 1 0. 0. 0. rect 0.113 0.113 0. 0. 26.5 26.5

h2: 2 0. 0. 0. rect 0.113 0.113 0. 0. 26.5 26.5

h3: 3 0. 0. 0. rect 0.113 0.113 0. 0. 26.5 26.5

h4: 4 0. 0. 0. rect 0.113 0.113 0. 0. 26.5 26.5

h5: 5 0. 0. 0. rect 0.113 0.113 0. 0. 26.5 26.5

h6: 6 0. 0. 0. rect 0.113 0.113 0. 0. 26.5 26.5

h7: 7 0. 0. 0. rect 0.113 0.113 0. 0. 26.5 26.5

h8: 8 0. 0. 0. rect 0.113 0.113 0. 0. 26.5
26.5
```

To take into account:

- The collimator could be either cylindrical or polygonal (flat faces).
In both cases, the holes are supposed to be parallel to detector plane.

- In case of polygonal collimator, the radius refers to the apothem.

- The number of holes is the total, considering all the angular
projections and rings. For each hole, the following information should
be introduced in rows (value previous to ':' is a label and it is not
read):

   - which detection element the whole projects to. In case one hole
projects to more than one detector element, it should be added twice,
once for each detector element using the relative position to that
detector element. To indicate that several holes projects to the same
detector element just assign them the same detector element index.

   - x, y, z coordinates (cm) of the centre of the hole in the collimator
plane. That is, 0 0 0 mean in the middle of the collimator face and in
the middle of the collimator wall. Coordinates x, z refer to spatial
position in the plane (horizontal and vertical respectivelly) and y is
the position of the hole in the wall thickness, so it should be kept
between +- wall\_thicknes/2.

       - rect or round depending if the hole has rectangular or round shape

       - hole dimensions (cm), x and z (horizontal and vertical)

       - x and z angles of the axis of the hole (in deg)

       - x and z acceptance angles from the hole axis

## Sign convention

![sign convention](sign-convention.png?raw=true)