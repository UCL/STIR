# Example files for running scatter estimation

Files made by Nikos Efthimou and fine-tuned by Kris Thielemans.<br>
Copyright University of Hull 2018-2019<br>
copyright University College London 2016, 2020<br>
Distributed under the Apache 2.0 License


This example is set-up using environment variables such that you can use these
files without modifying them if you want. Of course, you can also edit the
files to use explicit values instead.

The main file is `scatter_estimation.par`. It points to the other files using
the `scatter_pardir` environment variable.

Usage would be something like this
```sh
sino_input=myfile.hs
atnimg=myattenuationimage.hv
...

export sino_input atnimg ....
estimate_scatter scatter_estimation.par
```

See a full example in the `examples/Siemens-mMR` folder.
