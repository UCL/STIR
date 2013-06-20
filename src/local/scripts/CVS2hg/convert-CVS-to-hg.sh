#! /bin/bash
set -e
cd ~/devel/
rm -rf STIR-hg/*
mkdir -p STIR-hg/cvsroot
cp -rp cvsroot/CVSROOT cvsroot/parapet STIR-hg/cvsroot/

pushd STIR-hg/cvsroot/
mv parapet STIR
cd STIR
rm -rf doxygen doc
mv PPhead src
cd src
rm -rf   include/stir/local/ include/tomo/local/
# attempt to keep local files such that we detect revisions in moved files.
# However, that fails because of OBJFUNCbranch (can't remove the corresponding revisions yet by rm_revs.sh)

## remove all files that are still in local (i.e. not yet in an Attic
#find local -name Attic -prune -o -type f -name \*,v -exec rm {} \;
#find include/local -name Attic -prune -o -type f -name \*,v -exec rm {} \;
#mv local notyetindistro

# so, instead, we just remove them.
# consequence: files will have all history, but they will "exist" before a particular release
# e.g. hg up -r rel_1_00 extracts too many files
rm -rf include/local local

cd ..
find .   \( -name '*.h',v -o -name \*.hpp,v -o -name \*.cxx,v -o -name \*.txx,v -o -name \*.c,v -o -name \*.inl,v -o  -name Makefile\*,v -o -name \*.mk,v -o -name \*.cmake,v -o -name CMakeLists.txt,v -o -name \*.bat,v -o -name '*.txt',v -o -name \*sh,v -o -name \*.py,v -o -name \*.i,v -o -name \*tcl,v -o -name \*.rec,v -o -name \*.in,v -o -name \*.par,v -o -name Doxyfile\*,v -o -name ChangeLog,v -o -name zipit,v -o -name zipnewer,v -o -name PPlist,v -o  -name header_doc_sgl,v -o -name header_doc_sgl_edit,v -o -name show_header_sgl,v -o -name \*.\*s,v -o -name \*.htm,v -o -name \*.if,v -o -name \*.fdef,v -o -name \*.in,v -o -name \*.sh,v -o -name stir_subtract,v -o -name stir_divide,v -o -name count,v -o -name '*.dsw,v' -o -name '*.dsp,v' -o -name '*.sln,v' -o -name '*.vc*proj,v' -o -name '*.vcxproj.filters,v' -o -name \*htm,v -o -name \*sty,v -o -name \*tex,v -o -name Jam\* -o -name '*h[sv],v' -o -name \*inp,v \) -exec dos2unix {} \;
find .   \( -name '*.h',v -o -name \*.hpp,v -o -name \*.cxx,v -o -name \*.txx,v -o -name \*.c,v -o -name \*.inl,v -o  -name Makefile\*,v -o -name \*.mk,v -o -name \*.cmake,v -o -name CMakeLists.txt,v -o -name '*.txt',v  -o -name \*.i,v -o -name \*tcl,v -o -name \*.rec,v -o -name \*.in,v -o -name \*.par,v -o -name Doxyfile\*,v -o -name ChangeLog,v  -o -name \*.\*s,v -o -name \*.htm,v -o -name \*.if,v -o -name \*.fdef,v -o -name \*.in,v -o -name '*.dsw,v' -o -name '*.dsp,v' -o -name '*.sln,v' -o -name '*.vc*proj,v' -o -name '*.vcxproj.filters,v' -o -name \*htm,v -o -name \*sty,v -o -name \*tex,v -o -name Jam\* -o -name '*h[sv],v' -o -name '*.[sv],v' -o -name \*jpg,v -o -name \*png,v -o -name \*scn,v -o -name \*inp,v -o -name \*rtf,v \) -exec chmod -x {} \;

popd

cd STIR-hg
cvs -d  /home/kris/devel/STIR-hg/cvsroot/ checkout STIR
cd STIR

# handle ecat6. cut out revisions
rm -rf src/include/CTI  src/include/stir/CTI
for TAG in \
	rel_2_00_beta \
	trunk_after_merging_OBJFUNCbranch \
	OBJFUNC_before_merging_OBJFUNCbranch_to_trunk \
	trunk_before_merging_OBJFUNCbranch \
	OBJFUNC_rel_2_00_alpha \
        OBJFUNC_after_update_to_rel_1_40 \
        OBJFUNC_before_update_to_rel_1_40 \
        rel_1_40 \
        OBJFUNC_after_merge_for_patlak \
        OBJFUNC_before_merge_for_patlak \
        trunk_merge_to_OBJFUNC_for_patlak \
        OBJFUNC_updated_to_rel_1_40_beta \
        OBJFUNC_before_update_to_rel_1_40_beta \
        rel_1_40_beta \
        beforescannereffectiveringradius \
        rel_1_40_alpha \
        rel_MC_0_90 \
        rel_1_30 \
    OBJFUNC_updated_to_rel_1_30_beta \
    rel_1_30_beta \
    beforevectorwithcapacity \
    rel_1_20 \
    beforeOBJFUNCbranch \
    rel_1_11 \
    rel_1_10 \
    beforegeneralsymmetryloops \
    rel_1_00 \
    before_stir \
    rel_0_93 \
    rel_0_92_patches \
    rel_0_92_patched;
do
  for f in ecat6_types.h ecat6_utils.h stir_ecat6.h; do
    cvs tag -d  $TAG src/include/stir/IO/$f
  done
done
cvs admin -o 1.13.2.1:1.13.2.2  src/include/stir/IO/ecat6_utils.h
cvs admin -o 1.9.2.1 src/include/stir/IO/ecat6_types.h
cvs admin -o 1.13.2.1  src/include/stir/IO/stir_ecat6.h
  for f in ecat6_types.h ecat6_utils.h stir_ecat6.h; do
    cvs admin -o ::rel_2_00 src/include/stir/IO/$f
  done

cvs rtag -d 'before_VC_project_conversion_to_2010' STIR
#after_moving_modelling_to_global

# change dates in documentation *rtf such that they coincide with actual release

#  *rtf change date to avoid fixups for rel_0_92 and rel_1_00
for f in ~/devel/STIR-hg/cvsroot/STIR/documentation/Attic/*rtf,v; do
   if [ ! -r $f.org ]; then
       mv $f $f.org 
   fi
   sed -e 's/date\t2001.06.15/date\t2001.05.15/' -e 's/date\t2003.06.02.14.38/date\t2001.12.20.14.38/' $f.org > $f
done
	


cvs rtag -dB tag STIR
# remove these to avoid problems with fixup
cvs rtag -d rel_1_40_beta STIR
cvs rtag -d OBJFUNC_updated_to_rel_1_40_beta STIR
# a problem with test_VAXfloat.cxx (tag applied to listmode_buildblock/lib.mk a few weeks later?)
cvs rtag -d rel_2_20_alpha STIR
cvs rtag -d beforescannereffectiveringradius STIR
cvs rtag -d beforevectorwithcapacity STIR
cvs rtag -d beforegeneralsymmetryloops STIR

# get rid of RS_AK branch stuff
cvs rtag -d SHAPE3D_UPDATES_1_00 STIR
for f in ~/devel/STIR-hg/cvsroot/STIR/documentation/contrib/Shape3D_enhancements_RS_AK/*,v; do
   if [ ! -r $f.org ]; then
       mv $f $f.org 
   fi
   sed '-es/:1.1.3.1/:1.1/' '-es/RS_AK:1.1.3//' '-es/branch.*1.1.3;//' '-es/author kris/author schmidtlein/' $f.org > $f;
done
cvs rtag -dB RS_AK STIR
cd src/
for f in ../documentation/contrib/Shape3D_enhancements_RS_AK/*.* Shape_buildblock/lib.mk Shape_buildblock/Box3D.cxx Shape_buildblock/EllipsoidalCylinder.cxx include/stir/Shape/Box3D.h include/stir/Shape/EllipsoidalCylinder.h test/test_ROIs.cxx; do cvs admin -o 1.1.3.1 $f;done
cd ../
for f in ~/devel/STIR-hg/cvsroot/STIR/src/Shape_buildblock/Box3D.cxx,v ~/devel/STIR-hg/cvsroot/STIR/src/include/stir/Shape/Box3D.h,v; do
   if [ ! -r $f.org ]; then
       mv $f $f.org 
   fi
   sed -e 's/\(date\t2008.05.20.19.[23].*\)kris/\1schmidtlein/' $f.org > $f
done

# remove some branch revision which shouldn't be there anyway
cvs admin -o 1.4.2.1 src/include/stir/analytic/FBP3DRP/FBP3DRPReconstruction.h 
cvs admin -o 1.3.2.1 src/analytic/FBP3DRP/ColsherFilter.cxx 
cvs admin -o 1.1.2.1 src/analytic/FBP3DRP/exe.mk 
cvs admin -o 1.1.2.1 src/analytic/FBP3DRP/lib.mk 

~/devel/hgroot/rm_revs_dir.sh src/include 2>&1 > ../cvs_manips.log 
~/devel/hgroot/rm_revs_dir.sh src/include/tomo 2>&1 >> ../cvs_manips.log 
~/devel/hgroot/rm_revs_dir.sh src/include/recon_buildblock 2>&1 >> ../cvs_manips.log 
~/devel/hgroot/rm_revs_dir.sh src/include/OSMAPOSL 2>&1 >> ../cvs_manips.log 


cd ..
# this used to fail. maybe it might work now.
# see https://groups.google.com/forum/?hl=en&fromgroups=#!topic/mercurial_general/7-ObkSj17qE
#hg convert --authormap ~/devel/hgroot/STIR-authors.txt --filemap ~/devel/hgroot/STIR-filemap.txt . ../STIR-hg


../cvs2hg/cvs2svn/cvs2hg --options=../hgroot/cvs2hg.options >& ../hgroot/cvs2hg.log


#cvs admin: /home/kris/devel/STIR-hg/cvsroot/STIR/src/include/recon_buildblock/Attic/Reconstruction.h,v: Revision 1.19 doesn't exist.
#cvs admin: RCS file for `Reconstruction.h' not modified.

cd main.hg
hg up -C OBJFUNCbranch
hg commit --close-branch -m"closing OBJFUNBbranch. It's long been merged." 
hg up -C default


# TODO fix keywords
# sed script changes "Copyright ... $Date$" to "Copyright ... 2009 ..."
# and removed lines with $Id$ and only $date$ and $Revision$

#sed -e's/\(Copyright.*\)\$Date$/\1\2/' -e '/\$Id$/d' -e'/ *\$Date$ *$/d' -e'/ *\$Revision$ *$/d' 

cd ..
