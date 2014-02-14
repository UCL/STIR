#! /bin/bash
set -e
cd ~/devel/
rm -rf STIR-hg/*

if [ -r ~/devel/hgroot/cvsroot/STIR ]; then
  cp -rp ~/devel/hgroot/cvsroot STIR-hg
  cd STIR-hg
  cvs -d  /home/kris/devel/STIR-hg/cvsroot/ checkout STIR
  cd STIR

else

mkdir -p STIR-hg/cvsroot
cp -rp cvsroot/CVSROOT cvsroot/parapet STIR-hg/cvsroot/

pushd STIR-hg/cvsroot/
mv parapet STIR
cd STIR
rm -rf doxygen doc
mv PPhead src
rm -rf   src/include/stir/local/ src/include/tomo/local/

# remove files that are already in Attic
rm src/local/howto/howto_compile_AVRWOI.txt,v
rm src/local/recon_test/input/POSSPS.par,v
rm src/local/reconstruction_data/params/966/rescale_atten.par,v
rm src/local/scripts/counts_in_images.sh,v
rm src/local/scripts/is_interfile.sh,v

# remove a few files that create trouble and we're not interested in anyway
rm src/local/test/test_NRVO.cxx,v
rm documentation/release_template.htm,v
rm src/local/*gangelis.mk,v
rm src/local/extra_dirs*.mk,v
rm src/local/config_*.mk,v
rm -rf src/local/reconstruction_data/params

# uninteresting stuff
rm src/local/recon_buildblock/old*cxx,v src/include/local/stir/recon_buildblock/old*,v src/include/local/recon_buildblock/Attic/old*,v
rm src/local/buildblock/cleanup966ImageProcessor.cxx,v src/include/local/stir/cleanup966ImageProcessor.h,v
rm -rf src/local/howto

cd src

# remove Numerical Recipes
rm local/buildblock/fft.cxx,v
# remove QHidac
rm -rf local/QH*  include/local/stir/QHidac include/local/tomo/QHidac
# remove  GE things
cd local
rm -rf recon_buildblock/BackProjectorByBinDistanceDriven.cxx,v recon_buildblock/ForwardProjectorByBinDistanceDriven.cxx,v ../include/local/stir/recon_buildblock/BackProjectorByBinDistanceDriven.h,v ../include/local/stir/recon_buildblock/ForwardProjectorByBinDistanceDriven.h,v IO/GE ../include/local/stir/IO/GE motion/ScatterSimulationByBinWithMotion.cxx,v ../include/local/stir/motion/ScatterSimulationByBinWithMotion.h,v motion_utilities/fwd_image_and_fill_missing_data.cxx,v motion_utilities/simulate_scatter_with_motion.cxx,v scatter/*,v scatter_buildblock/*,v scatter_buildblock/*two_*,v ../include/local/stir/DoubleScatterEstimationByBin.h,v  utilities/Hounsfield2mu.cxx,v reconstruction_data/ ../include/local/stir/IO/Attic/ProjDataVOLPET.h,v ../include/local/stir/IO/Attic/niff.h,v IO/Attic/ProjDataVOLPET* IO/Attic/niff*
cd ..
# remove old/irrelevant scripts
cd local/scripts
rm -rf CalibrateImage966,v change_voxel_sizes_in_hv.sh,v find_normfile.sh,v get_* Makefile,v newpostproc.sh,v print_original_voxel_size_for_cti_zoom.sh,v process_966_transmission.sh,v recon* Recons* rmtags,v up-comp-debug.sh,v zip* scatter GE *total* *ecat* cvs2cl.pl,v check* cloc* compare* construct* correct_voxel* MCdistrib.sh,v *counts* PPlist* q* remove_q* add_pre* rename.sh,v set* copy* extract* header_doc* is_norm* MetaIO* is_Meta* precorrect* run_splines* show_header* Evaluation how_many* estimate_scatter.sh,v get_num_frames* Attic
cd ../..
cd ..

# handle ecat6. cut out revisions
rm -rf src/include/CTI  src/include/stir/CTI

# attempt to keep local files such that we detect revisions in moved files.


## remove all files that are still in local (i.e. not yet in an Attic
#find local -name Attic -prune -o -type f -name \*,v -exec rm {} \;
#find include/local -name Attic -prune -o -type f -name \*,v -exec rm {} \;
#mv local notyetindistro

# so, instead, we just remove them.
# consequence: files will have all history, but they will "exist" before a particular release
# e.g. hg up -r rel_1_00 extracts too many files
#rm -rf include/local local

cd ..
find .   \( -name '*.h',v -o -name \*.hpp,v -o -name \*.cxx,v -o -name \*.txx,v -o -name \*.c,v -o -name \*.inl,v -o  -name Makefile\*,v -o -name \*.mk,v -o -name \*.cmake,v -o -name CMakeLists.txt,v -o -name \*.bat,v -o -name '*.txt',v -o -name \*sh,v -o -name \*.py,v -o -name \*.i,v -o -name \*tcl,v -o -name \*.rec,v -o -name \*.in,v -o -name \*.par,v -o -name Doxyfile\*,v -o -name ChangeLog,v -o -name zipit,v -o -name zipnewer,v -o -name PPlist,v -o  -name header_doc_sgl,v -o -name header_doc_sgl_edit,v -o -name show_header_sgl,v -o -name \*.\*s,v -o -name \*.htm,v -o -name \*.if,v -o -name \*.fdef,v -o -name \*.in,v -o -name \*.sh,v -o -name stir_subtract,v -o -name stir_divide,v -o -name count,v -o -name '*.dsw,v' -o -name '*.dsp,v' -o -name '*.sln,v' -o -name '*.vc*proj,v' -o -name '*.vcxproj.filters,v' -o -name \*htm,v -o -name \*sty,v -o -name \*tex,v -o -name Jam\* -o -name '*h[sv],v' -o -name \*inp,v \) -exec dos2unix {} \;
find .   \( -name '*.h',v -o -name \*.hpp,v -o -name \*.cxx,v -o -name \*.txx,v -o -name \*.c,v -o -name \*.inl,v -o  -name Makefile\*,v -o -name \*.mk,v -o -name \*.cmake,v -o -name CMakeLists.txt,v -o -name '*.txt',v  -o -name \*.i,v -o -name \*tcl,v -o -name \*.rec,v -o -name \*.in,v -o -name \*.par,v -o -name Doxyfile\*,v -o -name ChangeLog,v  -o -name \*.\*s,v -o -name \*.htm,v -o -name \*.if,v -o -name \*.fdef,v -o -name \*.in,v -o -name '*.dsw,v' -o -name '*.dsp,v' -o -name '*.sln,v' -o -name '*.vc*proj,v' -o -name '*.vcxproj.filters,v' -o -name \*htm,v -o -name \*sty,v -o -name \*tex,v -o -name Jam\* -o -name '*h[sv],v' -o -name '*.[sv],v' -o -name \*jpg,v -o -name \*png,v -o -name \*scn,v -o -name \*inp,v -o -name \*rtf,v \) -exec chmod -x {} \;

find .   \( -name '*.sh,v' -o -name \*.tcl,v -o -name zip\*,v  \) -exec chmod +x {} \;


popd

cd STIR-hg
cvs -d  /home/kris/devel/STIR-hg/cvsroot/ checkout STIR
cd STIR

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

#  *rtf change date to avoid fixups for rel_0_92, rel_1_00 (and rel_1_20?)
for f in ~/devel/STIR-hg/cvsroot/STIR/documentation/Attic/*rtf,v; do
   if [ ! -r $f.org ]; then
       mv $f $f.org 
   fi
   sed -e 's/date\t2001.06.15/date\t2001.05.15/' -e 's/date\t2003.06.02.14.38/date\t2001.12.20.14.38/'  $f.org > $f
  #-e 's/date\t2004.04.02.17.13.49/date\t2004.04.01.00.13.49/' not necessary?
done

cvs rtag -dB tag STIR
# remove these to avoid problems with fixup
cvs rtag -d rel_1_40_beta STIR
cvs rtag -d OBJFUNC_updated_to_rel_1_40_beta STIR

cvs rtag -d OBJFUNC_before_merging_OBJFUNCbranch_to_trunk STIR
cvs rtag -d OBJFUNC_before_update_to_rel_1_40_beta STIR
cvs rtag -d OBJFUNC_after_update_to_rel_1_40 STIR
cvs rtag -d OBJFUNC_updated_to_rel_1_30_beta STIR
cvs tag -r 1.29 -F rel_2_30_beta src/local/scripts/distrib.sh
cvs tag -r 1.7 -F rel_1_30 src/local/scripts/distrib.sh

# a problem with test_VAXfloat.cxx (tag applied to listmode_buildblock/lib.mk a few weeks later?)
cvs rtag -d rel_2_20_alpha STIR
cvs rtag -d beforescannereffectiveringradius STIR
cvs rtag -d beforevectorwithcapacity STIR
cvs rtag -d beforegeneralsymmetryloops STIR
for t in \
 OBJFUNC_after_merge_for_patlak \
 OBJFUNC_before_merge_for_patlak STIR \
 OBJFUNC_before_update_to_rel_1_40 \
  OBJFUNC_rel_2_00_alpha \
  trunk_merge_to_OBJFUNC_for_patlak \
  after_moving_modelling_to_global \
  rel_2_10_alpha \
  rel_MC_0_90 \
  before_ExamInfo ;
do
  cvs rtag -d $t STIR
done

cp -rp 	~/devel/STIR-hg/cvsroot ~/devel/hgroot

fi

# change a few dates to get all moves before rel_1_00 in one commit
for f in ~/devel/STIR-hg/cvsroot/STIR/src/LICENSE.txt,v ~/devel/STIR-hg/cvsroot/STIR/src/VERSION.txt,v; do
   if [ ! -r $f.org ]; then
       mv $f $f.org 
   fi
   sed -e 's/date\t2001.12.20.21.17.17/date\t2001.12.20.21.21.48/' -e 's/date\t2001.12.20.20.04.08/date\t2001.12.20.20.21.22.49/' $f.org > $f
done
# and get rid of one irrelevant revision
cvs admin -o 1.3 src/test/test_display.cxx

# change date of move (commit was one year too late)
f=~/devel/STIR-hg/cvsroot/STIR/src/local/listmode_utilities/Attic/lm_to_projdata.cxx,v
   if [ ! -r $f.org ]; then
       mv $f $f.org 
   fi
   sed -e 's/date\t2005.02.24.13.56.00/date\t2004.03.19.14.56.00/' $f.org >$f

 for n in CListRecordECAT962.h CListRecordECAT966.h; do
   f=~/devel/STIR-hg/cvsroot/STIR/src/include/local/stir/listmode/Attic/${n},v
   if [ ! -r $f.org ]; then
       mv $f $f.org 
   fi
   sed -e 's/date\t2005.03.08.15.24.09/date\t2004.03.03.12.08.50/' -e 's/date\t2005.12.05.10.27.48/date\t2004.03.03.12.08.50/' $f.org >$f
done
   cvs admin -m1.2:"moved to global distro" src/include/local/stir/listmode/CListRecordECAT962.h
   cvs admin -m1.3:"moved to global distro" src/include/local/stir/listmode/CListRecordECAT966.h

   # now need to remove tags as well to take this into account
   for t in OBJFUNC_updated_to_rel_1_40_beta \
        OBJFUNC_before_update_to_rel_1_40_beta \
        rel_1_30 \
        OBJFUNC_updated_to_rel_1_30_beta \
        rel_1_30_beta \
        rel_1_20; do
     cvs tag -d $t src/local/listmode_utilities/lm_to_projdata.cxx src/include/local/stir/listmode/CListRecordECAT962.h src/include/local/stir/listmode/CListRecordECAT966.h
   done

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

# fix problem with Reconstruct.h being reinstated as different file
mv ~/devel/STIR-hg/cvsroot/STIR/src/include/Attic/Reconstruction.h,v ~/devel/STIR-hg/cvsroot/STIR/src/include/Attic/XXXReconstruction.h,v

# find copies. Important: these have to be in chronological order
~/devel/hgroot/rm_revs_dir.sh src/include > ../cvs_manips.log  2>&1
~/devel/hgroot/rm_revs_dir.sh src/include/tomo  >> ../cvs_manips.log  2>&1
~/devel/hgroot/rm_revs_dir.sh src/include/recon_buildblock  >> ../cvs_manips.log   2>&1
~/devel/hgroot/rm_revs_dir.sh src/include/tomo/recon_buildblock  >> ../cvs_manips.log  2>&1
~/devel/hgroot/rm_revs_dir.sh src/include/OSMAPOSL 2>&1 >> ../cvs_manips.log 
~/devel/hgroot/rm_revs_dir.sh src/include/local >> ../cvs_manips.log  2>&1
~/devel/hgroot/rm_revs_dir.sh src/include/local/tomo >> ../cvs_manips.log  2>&1
~/devel/hgroot/rm_revs_dir.sh src/include/local/tomo/recon_buildblock >> ../cvs_manips.log  2>&1
~/devel/hgroot/rm_revs_dir.sh src/include/local/tomo/Shape >> ../cvs_manips.log  2>&1
~/devel/hgroot/rm_revs_dir.sh src/include/local/tomo/eval_buildblock >> ../cvs_manips.log  2>&1
#~/devel/hgroot/rm_revs_dir.sh src/include/local/stir >> ../cvs_manips.log  2>&1
#~/devel/hgroot/rm_revs_dir.sh src/include/local/stir/recon_buildblock >> ../cvs_manips.log  2>&1
#~/devel/hgroot/rm_revs_dir.sh src/include/local/stir/modelling >> ../cvs_manips.log  2>&1
find src/include/local/stir  -name CVS -prune  -o -type d -exec ~/devel/hgroot/rm_revs_dir.sh {} \;  >> ../cvs_manips.log  2>&1


~/devel/hgroot/rm_revs_dir.sh src/local/listmode  >> ../cvs_manips.log  2>&1
# need to move this out of the way now
#mv ~/devel/STIR-hg/cvsroot/STIR/src/local/listmode/Attic ~/devel/STIR-hg/cvsroot/STIR/src/local/listmode/Attic.org


find src/local  -name CVS -prune  -o -type d -exec ~/devel/hgroot/rm_revs_dir.sh {} \;  >> ../cvs_manips.log  2>&1

# move back
#mv ~/devel/STIR-hg/cvsroot/STIR/src/local/listmode/Attic.org ~/devel/STIR-hg/cvsroot/STIR/src/local/listmode/Attic

# ecat utilities moved
~/devel/hgroot/rm_revs_dir.sh src/utilities 2>&1 >> ../cvs_manips.log 

mv ~/devel/STIR-hg/cvsroot/STIR/src/include/Attic/XXXReconstruction.h,v ~/devel/STIR-hg/cvsroot/STIR/src/include/Attic/Reconstruction.h,v

pushd ~/devel/STIR-hg/cvsroot/STIR/src/
# get rid of some makefiles that seem to create trouble
#rm -f local/Attic/Makefile* local/*/Attic/Makefile* listmode_utilities/Attic/Makefile*
# duplicate
rm -f include/local/Attic/BackProjectorByBinUsingSquareProjMatrixByBin.h,v include/local/stir/Attic/BackProjectorByBinUsingSquareProjMatrixByBin.h,v
# erroneous
rm -f include/local/stir/Attic/IR_filters.*~,v
# trouble makers
find . -name \*.dsp,v -exec rm {} \;
find . -name \*.dsw,v -exec rm {} \;
popd


cd ..
# this used to fail. it might work now.
# see https://groups.google.com/forum/?hl=en&fromgroups=#!topic/mercurial_general/7-ObkSj17qE
#hg convert --authormap ~/devel/hgroot/STIR-authors.txt --filemap ~/devel/hgroot/STIR-filemap.txt . ../STIR-hg


~/devel/cvs2hg/cvs2svn/cvs2hg --options=../hgroot/cvs2hg.options >& ../hgroot/cvs2hg.log

# now convert to git
mkdir STIRfrommerc;cd STIRfrommerc
git init;~/devel/fast-export/hg-fast-export.sh -r ../main.hg/
git checkout master
cd ..



# don't bother closing the branch. it creates an ugly commit right at the top.
#cd main.hg
#hg up -C OBJFUNCbranch
#hg commit --close-branch -m"closing OBJFUNCbranch. It's long been merged and no longer active." 
#hg up -C default
#cd ..

# ../cvs2svn-trunk/cvs2git --options=../hgroot/cvs2git.options >& ../hgroot/cvs2git.log

#git init --bare STIR.git
#cd STIR.git
#cat ../cvs2git-tmp/git-blob.dat ../cvs2git-tmp/git-dump.dat | git fast-import
#git branch -D TAG.FIXUP
#gitk --all

#git checkout -b open rel_1_00
#git tag release_1_00
#git checkout -b open rel_0_93
#git tag release_0_93
#git merge --no-commit rel_1_00

#(Recommended) To get rid of unnecessary tag fixup branches, run the contrib/git-move-refs.py script from within the git repository.

# TODO fix keywords
# sed script changes "Copyright ... 2009" to "Copyright ... 2009 ..."
# and remove lines with $Id$ and only $date$ and $Revision$

#sed -e's/\(Copyright.*\)\$Date$/\1\2/' -e '/\$Id$/d' -e'/ *\$Date$ *$/d' -e'/ *\$Revision$ *$/d' 

cd ..
