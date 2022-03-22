#! /bin/bash
# A script to package a new distribution of STIR.
# It is probably specific to the set-up of files on Kris Thielemans' computer,
# although should need only minor tweaking for others.
# It is not documented though and possibly unsafe.
# Use with care!
#
# You would use this in bash for instance like
# VERSION=3.1 make_distribution.sh
# Check list of variables below for configuration options

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details
#
# Copyright 2004-2011, Hammersmith Imanet Ltd
# Copyright 2011-2013, Kris Thielemans
# Copyright 2014-2015,2019,2020 University College London


# set default for variables.
# A lot of these are for being able to do the processing in stages
# (e.g. for when something went wrong)
: ${do_update:=0}
: ${do_version:=1}
: ${do_license:=1}
: ${do_ChangeLog:=1}
: ${do_doc:=1}
: ${do_doxygen:=1}
: ${do_git_commit:=1}
: ${do_zip_source:=1}
: ${do_recon_test_pack:=1}
: ${do_transfer:=1}

# only enable this when non-beta version
: ${do_website_final_version:=0}
: ${do_website_sync:=0}

set -e
: ${VERSION:=4.1.0}
: ${TAG:=rel_${VERSION}}


: ${REPO:=git@github.com:UCL/STIR} #=~/devel/UCL_STIR}
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


read -p "Did you update CMakeLists.txt, version numbers in \*tex files, documentation/history.htm, .zenodo.json? (press Ctrl-C if not)"

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
trap "echo ERROR in updating VERSION.txt" ERR
echo $VERSION > VERSION.txt
git add VERSION.txt
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
  find . -path .git -prune \
     -o -name "*[xhlkc]" -type f  -print | grep -v .git| grep -v maintenance | xargs grep -l PARAPET-license  >>LICENSE.txt 
  git add LICENSE.txt
fi

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
    mkdir -p ${DISTRIB}/build/STIR_${VERSION}
    pushd ${DISTRIB}/build/STIR_${VERSION}
    cmake -DGRAPHICS=None ${DISTRIB}/STIR
    echo "CMake OK"
    make RUN_DOXYGEN > ${DISTRIB}/doxygen.log 2>&1
    mkdir -p ${DISTRIB}/STIR/documentation/doxy
    #mv html ${DISTRIB}/STIR/documentation/doxy/
    cd ${DISTRIB}/STIR/documentation/doxy/
    if test -L html; then
        rm html
    fi
    ln -s ${DISTRIB}/build/STIR_${VERSION}/html
    popd
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

if [ $do_git_commit = 1 ]; then
    trap "echo ERROR with git" ERR
    cd ${DISTRIB}/STIR
    if git diff --cached --exit-code; then
        echo "No changes staged. git commit not called."
    else
        git commit  -m "updated VERSION.txt etc for release of version $VERSION"
    fi
    if git rev-parse "$TAG" >/dev/null 2>&1; then
        echo "git tag $TAG exists!. Removing"
        git tag -d $TAG
        # git tag -d stir_$TAG
    fi
    git tag -a $TAG -m "version $VERSION";
    # git tag -a stir_$TAG -m "version $VERSION";
else
    echo "no git commit/tagging"
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
    #ln -s .htaccessSF .htaccess
    cd ../documentation
    rm STIR_doc.zip
    ln -s STIR_doc_${VERSION}.zip STIR_doc.zip 
    rm -fr doxy
    unzip -u STIR_doc
    rm -rf contrib
    mv STIR/documentation/* .
    rmdir STIR/documentation
    rmdir STIR
    cd ..
fi

if [ $do_website_sync = 1 ]; then
    cd $destination
    ./sync-to-sf.sh --del
fi

echo "still do 'git push; git push --tags'"
echo "if not beta, did you run with 'do_website_final_version=1'?"
