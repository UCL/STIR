# Example files for running scatter estimation for the Siemens mMR

Files made by Nikos Efthimou and fine-tuned by Kris Thielemans.<br>
Copyright University of Hull 2018-2019<br>
copyright University College London 2016, 2020<br>
Distributed under the Apache 2.0 License

These files are almost identical to those in
[examples/samples/scatter_estimation_par_files/](../../samples/scatter_estimation_par_files/README.md),
see there for some more information.

Currently the only difference are the lower values for
```
maximum scatter scaling factor := .5
minimum scatter scaling factor := 0.1
```

These have been shown to work better for mMR data, see e.g.
[STIR issue #1163](https://github.com/UCL/STIR/issues/1163).
