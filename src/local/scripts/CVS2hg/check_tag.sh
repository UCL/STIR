#! /bin/bash
# check if files in tagged CVS and git are the same

set -e
tag=$1
cd ~/devel/STIR-hg
rm -f dif.log

rm -rf STIR
cvs -d  /home/kris/devel/hgroot/cvsroot/ checkout -kk -r $tag -P STIR > /dev/null 2>&1
cd STIR
cd src
set +e
# known cases of DOS endings in CVS version
#dos2unix Jamrules Doxyfile recon_test/input/*f test/modelling/input/*[fn] test/numerics/Makefile_BSpline_timings ../doximages/*eps
cd ../../STIRfrommerc
git checkout $tag
diff -r -u -x .git -x CVS . ../STIR > ../dif.log
echo output in `pwd`/../dif.log
