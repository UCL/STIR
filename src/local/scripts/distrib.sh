#! /bin/sh
do_update=1
do_license=1
do_ChangeLog=1
do_doc=1
do_zip_source=1
do_recon_test_pack=1
do_transfer=0

set -e
# rsync of website note: stalls on gluon,wren,hurricane, but works fine from shark
VERSION=1.3
CHECKOUTOPTS="-r rel_1_30"
cd $WORKSPACE/../..
DISTRIB=`pwd`/distrib
LLN=${DISTRIB}/../lln
WORKSPACE=${DISTRIB}/parapet/PPhead 

destination=krthie@shell.sf.net:stir/htdocs/
RSYNC_OPTS=
#destination=web@wren:htdocs/STIR/
#RSYNC_OPTS=--rsync-path=/home/kris/bin/rsync 
mkdir -p ${DISTRIB}
cd ${DISTRIB}

echo "Did you  make ${DISTRIB}/release_${VERSION}.htm ?"

echo LLN stuff (still by hand)
if [ ! -r ${LLN}/ecat.tar.gz ]; then
  echo Need LLN in ${LLN}
  exit 1
fi
#tar --exclude VC --exclude CVS -czf ecat.tar.gz ecat

if [ ! -r parapet ]; then
  cvs -d parapet:/usr/local/cvsroot $CHECKOUTOPTS checkout parapet
  cd parapet
else
  cd parapet
  if [ $do_update = 1 ]; then
    cvs up -dP
  fi
fi
rm -f STIR
ln -s PPhead STIR
cd PPhead

# update VERSION.txt
echo $VERSION > VERSION.txt
cvs commit -m "- updated for release of version $VERSION" VERSION.txt

# update LICENSE.txt
if [ $do_license = 1 ]; then
  cd $WORKSPACE
  # put version in there
  cat LICENSE.txt | \
  sed "s/Licensing information for STIR .*/Licensing information for STIR $VERSION/" \
  > tmp_LICENSE.txt
  # remove list of files at the end (dangerous: relies on the text in the file)
  END_STRING="----------------------------------------------------"
  AWK_PROG="{ if( \$1 ~ \"$END_STRING\") {
                 exit 0;
            } else {
              print \$0
            }
          }"
  awk "$AWK_PROG" tmp_LICENSE.txt > LICENSE.txt
  echo $END_STRING >> LICENSE.txt
  #then add new list on again
  find . -path ./local -prune -path ./include/local -prune \
     -o -name "*[xhlkc]"  -print|grep -v CVS | xargs grep -l PARAPET >>LICENSE.txt 
  cvs commit  -m "- updated for release of version $VERSION" LICENSE.txt
fi

# make ChangeLog file
if [ $do_ChangeLog = 1 ]; then
  cd $WORKSPACE
  # maybe use --accum
  mv local xxlocal
  cvs2cl.pl -I 'xxlocal/' -I 'include/local'  --no-indent -F trunk
  mv xxlocal local
  cp ChangeLog ${DISTRIB}
fi

if [ $do_doc = 1 ]; then
  cd $WORKSPACE
  # make doxygen
  doxygen
  # make documentation PDFs BY HAND
  cd ../documentation
  make
  zip -ur ${DISTRIB}/STIR_doc_${VERSION}.zip *.pdf  doxy
fi

if [ $do_zip_source ]; then
  cd ${DISTRIB}
  rm -f parapet/all.zip parapet/VCprojects.zip
  zipit --distrib
  zipproj --distrib
  mv parapet/VCprojects.zip VCprojects_${VERSION}.zip 
  mv parapet/all.zip STIR_${VERSION}.zip 
fi

if [ $do_recon_test_pack = 1 ]; then
  cd ${DISTRIB}/parapet/
  #rm -rf recon_test_pack/CVS
  zip -ulr ../recon_test_pack_${VERSION}.zip recon_test_pack -i CVS/ CVS/*
  #tar zcvf ../recon_test_pack_${VERSION}.tar.gz recon_test_pack
fi

if [ $do_transfer = 1 ]; then
  chmod go+r *${VERSION}* ChangeLog
  chmod go-wx *${VERSION}* ChangeLog

  # put it all there
  rsync --progress -uavz ${RSYNC_OPTS}  ${LLN}/ecat/VC/ecat.dsp ${LLN}/ecat.tar.gz \
    STIR_${VERSION}.zip VCprojects_${VERSION}.zip \
    recon_test_pack_${VERSION}.zip \
    ${destination}registered
  rsync --progress -uavz ${RSYNC_OPTS} ChangeLog release_${VERSION}.htm STIR_doc_${VERSION}.zip  \
    ${destination}documentation
fi

exit 
# remote
VERSION=1.3
echo EDIT documentation/history.htm
rm  recon_test_pack.tar.gz STIR.zip VCprojects.zip recon_test_pack.zip 
ln -s STIR_${VERSION}.zip STIR.zip 
ln -s VCprojects_${VERSION}.zip  VCprojects.zip
#ln -s recon_test_pack_${VERSION}.tar.gz  recon_test_pack.tar.gz 
ln -s recon_test_pack_${VERSION}.zip recon_test_pack.zip
cd ../documentation
rm STIR_doc.zip
ln -s STIR_doc_${VERSION}.zip STIR_doc.zip 
rm -fr doxy
unzip -u STIR_doc

