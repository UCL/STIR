#! /bin/bash
set -e
orgdir=$1
cvsdir=~/devel/STIR-hg/cvsroot/STIR/
cd $cvsdir
for f in ${orgdir}/Attic/*.*[xhl],v; do
 if [ -r $f ]; then
   name=`basename $f`
   numfound=`find . -type f -name ${name}|wc -l`
   if [ $numfound -gt 1 ]; then
       find . -type f -name ${name} -exec  ~/devel/hgroot/rm_revs.sh ./$f {} \;
   else
       echo "$name not found elsewhere ($f)"
   fi
fi
done

   
