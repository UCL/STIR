#! /bin/sh
set -e
# rsync of website note: stalls on gluon,wren,hurricane, but works fine from shark
VERSION=1.3
cd $WORKSPACE/../..
DISTRIB=`pwd`/distrib

destination=krthie@shell.sf.net:stir/htdocs/
RSYNC_OPTS=--progress
#destination=web@wren:htdocs/STIR/
#RSYNC_OPTS=  --progress --rsync-path=/home/kris/bin/rsync 
mkdir -p ${DISTRIB}
cd ${DISTRIB}
echo somehow make release_${VERSION}.htm

echo LLN stuff
# ecat.tar.gz
#tar --exclude VC --exclude CVS -czf ecat.tar.gz ecat
if [ ! -r parapet ]; then
  cvs -d parapet:/usr/local/cvsroot checkout parapet
  cd parapet
else
  cd parapet
  cvs up -dP
fi
rm -f STIR
ln -s PPhead STIR
cd PPhead

# update VERSION.txt
echo $VERSION > VERSION.txt
cvs commit -m "- updated for release of version $VERSION" VERSION.txt
# update LICENSE.txt
#check directories etc
#first remove list of files at the end
#then
# find . -path ./local -prune -path ./include/local -prune -name "*[xhlc]" -o -print|grep -v CVS | grep -v .dsp| xargs grep -l PARAPET >>LICENSE.txt 
#cvs commit  -m "- updated for release of version $VERSION" LICENSE.txt

# make ChangeLog file
# maybe use --accum
mv local xxlocal
cvs2cl.pl -I 'xxlocal/' -I 'include/local'  --no-indent -F trunk
mv xxlocal local
cp ChangeLog ${DISTRIB}

# make doxygen
doxygen
# make documentation PDFs BY HAND
cd ../documentation
make
zip -ur ${DISTRIB}/STIR_doc_${VERSION}.zip *.pdf  doxy

cd ${DISTRIB}
rm -f parapet/all.zip parapet/VCprojects.zip
WORKSPACE=${DISTRIB}/parapet/PPhead zipit --distrib
WORKSPACE=${DISTRIB}/parapet/PPhead zipproj --distrib
mv parapet/VCprojects.zip VCprojects_${VERSION}.zip 
mv parapet/all.zip STIR_${VERSION}.zip 


# recon_test_pack
 cd parapet/
#rm -rf recon_test_pack/CVS
zip -ulr ../recon_test_pack_${VERSION}.zip recon_test_pack -i CVS/ CVS/*
#tar zcvf ../recon_test_pack_${VERSION}.tar.gz recon_test_pack
cd ..

chmod go+r *${VERSION}* ChangeLog
chmod go-wx *${VERSION}* ChangeLog

# put it all there
rsync --progress -uavz ${RSYNC_OPTS}  ~/lln/ecat/VC/ecat.dsp ~/lln/ecat.tar.gz \
    STIR_${VERSION}.zip VCprojects_${VERSION}.zip \
    recon_test_pack_${VERSION}.zip \
    ${destination}registered
rsync --progress -uavz ${RSYNC_OPTS} ChangeLog release_${VERSION}.htm STIR_doc_${VERSION}.zip  \
    ${destination}documentation

exit 
# remote
VERSION=1.3
echo EDIT documentation/history.htm
rm  recon_test_pack.tar.gz STIR.zip VCprojects.zip recon_test_pack.zip 
ln -s STIR_${VERSION}.zip STIR.zip 
ln -s VCprojects_${VERSION}.zip  VCprojects.zip
ln -s recon_test_pack_${VERSION}.tar.gz  recon_test_pack.tar.gz 
ln -s recon_test_pack_${VERSION}.zip recon_test_pack.zip
cd ../documentation
rm STIR_doc.zip
ln -s STIR_doc_${VERSION}.zip STIR_doc.zip 
rm -fr doxy

unzip STIR_doc
 chmod go+r STIR_doc_${VERSION}.zip
R_doc.zip 
