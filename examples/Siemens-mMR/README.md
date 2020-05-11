# Example files to reconstruct data from the Siemens mMR.

These are currently not documented. You have to read them (and know some 
shell-scripting to get going).

The scripts use default names, which in most cases you can set before
on the command line. In bash or sh, you can do this as follows

```sh
sino_input=mysino.hs ECATNORM=mynorm.n.STIR howto_scatter_and_recon.sh
```

Naming of these variables is currently almost random, sorry.

The scripts don't do proper error handling. If they suddenly stop without diagnostics, go
and look for a log file.

The scripts need template files from this directory. By default
they assume they are located in `~/devel/STIR/examples/Siemens-mMR`.
If that is not the case, you can set the pardir variable as above, or
by doing for instance

```sh
pardir=~/STIR/examples/Siemens-mMR
export pardir
```

You can test the scripts by downloading the NEMA acquisitions from https://zenodo.org/record/1304454 and
use the `process_NEMA_data.sh` script.
```sh
curl -OL https://zenodo.org/record/1304454/files/
unzip NEMA_IQ.zip
cd NEMA_IQ
${pardir}/process_NEMA_data.sh
```
Final output will be called `final_activity_image_*.hv`

Good luck...

Kris Thielemans

