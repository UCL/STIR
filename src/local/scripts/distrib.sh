#! /bin/bash
: ${do_lln:=0}
: ${do_update:=0}
: ${do_version:=1}
: ${do_license:=1}
: ${do_ChangeLog:=1}
: ${do_doc:=1}
: ${do_doxygen:=1}
: ${do_zip_source:=1}
: ${do_recon_test_pack:=1}
: ${do_transfer:=1}

# only enable this when non-beta version
: ${do_website_final_version:=0}
: ${do_website_sync:=0}

set -e
: ${VERSION:=3.0}

: ${REPO:=~/devel/STIR -b open_source}
: ${CHECKOUTOPTS:=""}

: ${destination:=~/devel/STIR-website/}
: ${RSYNC_OPTS:=""}

: ${DISTRIB:=~/devel/STIRdistrib}

# disable warnings as we currently get rid of any existing zip files
# reasons:
# - this will make sure we do not have files that are removed in the distro in the zip file
# - zip -u returns funny error code when updating a zip file

#if [ $do_doc = 1 -a -r ${DISTRIB}/STIR_doc_${VERSION}.zip ]; then
#  echo WARNING: updating existing zip file ${DISTRIB}/STIR_doc_${VERSION}.zip
#fi
#if [ $do_recon_test_pack = 1 -a -r ${DISTRIB}/recon_test_pack_${VERSION}.zip ]; then
#  echo "WARNING: updating existing zip file ${DISTRIB}/recon_test_pack_${VERSION}.zip"
#fi


mkdir -p ${DISTRIB}
cd ${DISTRIB}

trap "echo ERROR in git clone" ERR
if [ ! -r STIR ]; then
    git clone $CHECKOUTOPTS  $REPO STIR
    cd STIR
else
  if [ $do_update = 1 ]; then
    trap "echo ERROR in git checkout" ERR
    cd STIR
    git pull
    git checkout  $CHECKOUTOPTS
  else
    cd STIR
  fi
fi

# update VERSION.txt
if [ $do_version = 1 ]; then
echo "updating VERSION.txt"
echo "TODO update PROJECT_NUMBER in Doxyfile"
trap "echo ERROR in updating VERSION.txt" ERR
echo $VERSION > VERSION.txt
fi

# update LICENSE.txt
if [ $do_license = 1 ]; then
  echo "updating LICENSE.txt"
  trap "echo ERROR in updating LICENSE.txt" ERR
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
  rm tmp_LICENSE.txt
  echo $END_STRING >> LICENSE.txt
  #then add new list on again
  find . -path ./local -prune -path ./include/local -prune -path ./include/stir/local -prune -path .git -prune \
     -o -name "*[xhlkc]" -type f  -print | xargs grep -l PARAPET |grep -v 'local/' >>LICENSE.txt 
fi

#git commit  -m "updated VERSION.txt etc for release of version $VERSION"

# make ChangeLog file
if [ $do_ChangeLog = 1 ]; then
  trap "echo ERROR in updating ChangeLog" ERR
  echo Do ChangeLog
  git log  --pretty=format:'-------------------------------%n%cD  %an  %n%n%s%n%b%n' --name-only > ${DISTRIB}/ChangeLog
fi

if [ $do_doc = 1 ]; then
  echo "Making doc"
  trap "echo ERROR in updating doc" ERR
  cd src
  # make doxygen
  if [ $do_doxygen = 1 ]; then
    PATH=$PATH:/cygdrive/c/Program\ Files/GPLGS:/cygdrive/d/Program\ Files/Graphviz2.26.3/bin
    echo "Running doxygen"
    doxygen > ${DISTRIB}/doxygen.log 2>&1
    mv dox.log ${DISTRIB}/
    echo "Done"
  fi
  cd ../documentation
  echo "make rtf->PDFs BY HAND"
  #cygstart /cygdrive/c/Program\ Files/Microsoft\ Office/OFFICE11/winword STIR_FBP3DRP.rtf 
  make
  pushd contrib/Shape3D_enhancements_RS_AK/
  pdflatex generate_image_upgrade.tex
  pdflatex generate_image_upgrade.tex
  rm -f *log *aux *out
  popd
  chmod go+x doxy
  chmod go+x doxy/html
  chmod -R go+r *
  cd ../..
  rm -f ${DISTRIB}/STIR_doc_${VERSION}.zip
  echo "zipping documentation"
  zip -rD ${DISTRIB}/STIR_doc_${VERSION}.zip STIR/documentation/*.rtf STIR/documentation/*.pdf STIR/documentation/*.htm STIR/documentation/doxy >/dev/null
  find STIR/documentation/contrib -type f | zip -@ ${DISTRIB}/STIR_doc_${VERSION}.zip 
fi

trap "echo ERROR after creating doc" ERR

if [ $do_zip_source = 1 ]; then
  echo Do zip source
  cd ${DISTRIB}
  #zipit --distrib > /dev/null
  #zipproj --distrib > /dev/null
  #mv parapet/VCprojects.zip VCprojects_${VERSION}.zip 
  #mv parapet/all.zip STIR_${VERSION}.zip 
  zip -rp STIR_${VERSION}.zip  STIR/src STIR/doximages STIR/examples STIR/scripts STIR/*.txt
fi

if [ $do_recon_test_pack = 1 ]; then
  cd ${DISTRIB}
  echo Do zip recon_test_pack
  rm -f recon_test_pack_${VERSION}.zip
  zip -r recon_test_pack_${VERSION}.zip STIR/recon_test_pack  > /dev/null
fi

if [ $do_transfer = 1 ]; then
  cd ${DISTRIB}
  chmod go+r *${VERSION}* ChangeLog
  chmod go-wx *${VERSION}* ChangeLog

  # put it all there
  rsync --progress -uavz ${RSYNC_OPTS} \
    STIR_${VERSION}.zip \
    recon_test_pack_${VERSION}.zip \
    ${destination}registered
  rsync --progress -uavz ${RSYNC_OPTS} \
    ChangeLog STIR_doc_${VERSION}.zip  \
    ${destination}documentation
  cd STIR/documentation
  rsync --progress -uavz ${RSYNC_OPTS} \
    *htm  \
    ${destination}documentation
fi

if [ $do_website_final_version = 1 ]; then
    cd $destination
    cd registered
    rm -f recon_test_pack.tar.gz STIR.zip VCprojects.zip recon_test_pack.zip 
    ln -s STIR_${VERSION}.zip STIR.zip 
    #ln -s VCprojects_${VERSION}.zip  VCprojects.zip
    #ln -s recon_test_pack_${VERSION}.tar.gz  recon_test_pack.tar.gz 
    ln -s recon_test_pack_${VERSION}.zip recon_test_pack.zip
    rm -f .htaccess
    ln -s .htaccessSF .htaccess
    cd ../documentation
    rm STIR_doc.zip
    ln -s STIR_doc_${VERSION}.zip STIR_doc.zip 
    rm -fr doxy
    unzip -u STIR_doc
    mv STIR/documentation/* .
    rmdir STIR/documentation
    rmdir STIR
    cd ..
fi

if [ $do_website_sync = 1 ]; then
    cd $destination
    ./sync-to-sf.sh --del
fi

