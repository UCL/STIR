# Example files for processing data from the GE Signa PET/MR
This should also work for data from GE PET/CT scanners using the RDF9 file format

Check the [howto.md](howto.md) for a description. In addition, there are some
(rather horrible) scripts that might help.

Currently incomplete is the attenuation processing, e.g. conversion from GE MRAC (PIFA) to mu-values,
or CTAC to mu. A basic script is provided for CTAC to mu, but this needs adaptation.
A script is provided to cope with flipping and rotation by registration once that's done.

As an example, we also provide complete processing of the data from the GE PET/MR
spatial calibration phantom (called VQC), available from [Zenodo](https://zenodo.org/record/3887517).
