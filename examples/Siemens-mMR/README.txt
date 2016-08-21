Example files to reconstruct data from the Siemens mMR.

These are currently not documented. You have to read them (and now some 
shell-scripting to get going).

The scripts use default names, which in most cases you can set before
on the command line. In bash or sh, you can do this as follows

sino_input=mysino.hs ECAT8NORM=mynorm.n.STIR howto_scatter_and_recon.sh

A few of the scripts need template files from this directory. By default
they assume they are located in ~/devel/STIR/examples/Siemens-mMR.
If that is not the case, you can set the pardir variable as above, or
by doing for instance

pardir=~/STIR/examples/Siemens-mMR
export pardir

Good luck...

Kris Thielemans

