#! /bin/bash
set -e
tag=$1
cd ~/devel/STIR-hg
rm -f dif.log
rm -rf parapet
if [ ! -r cvsrootnolocal ]; then
  rsync -auzv ~/devel/cvsroot/parapet ~/devel/STIR-hg/cvsrootnolocal/
 rsync -auzv ~/devel/cvsroot/CVSROOT ~/devel/STIR-hg/cvsrootnolocal/
  # remove these to save some time and less diffs
  rm -rf cvsrootnolocal/parapet/PPhead/local cvsrootnolocal/parapet/PPhead/include/local
fi

cvs -d  /home/kris/devel/STIR-hg/cvsrootnolocal/ checkout -kk -r $tag -P parapet
cd parapet
mv PPhead src
cd src
set +e
# known cases of DOS endings in CVS version
dos2unix Jamrules Doxyfile recon_test/input/*f test/modelling/input/*[fn] test/numerics/Makefile_BSpline_timings ../doximages/*eps
set -e
cd ../../mainkw.hg
hg up -r $tag
diff -r -u -x .hg -x CVS . ../parapet > ../dif.log
echo output in `pwd`/../dif.log
