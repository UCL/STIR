#! /bin/sh
# git-fame allows printing "loc" (lines of code), "coms" (number of commits) and "fils" (number of files)
# per contributor (using .mailmap)
#
# First use `git log --format='%aN <%aE>' | sort -u` to get list of authors and sort out
# duplicates via .mailmap.
#
# Suggested usage of git fame:
#  -w, --ignore-whitespace  Ignore whitespace when comparing the parent's
#                           version and the child's to find where the lines
#                           came from [default: False].
#  -M  Detect intra-file line moves and copies [default: False].
#  -C  Detect inter-file line moves and copies [default: False].
#
# In particular, `-C` takes ~30x longer to run, but a lot of this has happened in STIR
#
# Note that various files were apparently not in git before 2003 (rel_1_10) (they were in CVS).
# Also, some files did not have correct authorship.
# The output therefore needs manual ordering (see below)
git fame -wMC  --excl '\.(eps|root|ahv|hv|v|hs|s|scan|l|hdr|rtf|gz|if|pdf|safir|options|png|cls|sty)$|external_helpers|crystal_map|collimator.*txt|Doxyfile.in|LICENSE.txt|LICENSES' \
  | tee git-fame-output.txt
exit

# git history doesn't tell us accurately how many slocs the PARAPET people had
# just before 5.1 "git fame" reports the following for them
| Author                 |    loc |   coms
| Matthew Jacobson       |   1647 |     73 
| Alexey Zverovich       |   1270 |      6 
| Patrick Valente        |    239 |      2
| Claire Labbe           |    157 |     43 
| Damiano Belluzzo       |      0 |      3 
| Mustapha Sadki         |      0 |      3 
# KT suggests to correct the last to
| Claire Labbe           |    1000 (adding FBP3DRP)
| Mustapha Sadki         |     400 (adding raytracing)
| Damiano Belluzzo       |      60
| Darren Hague           |      50    

# Also, SPECTUB files were checked in by KT, but actually written by Carles Falcon.
# Finally PinholeSPECTUB is now attributed correctly (mostly to Carles)
# Just before release 5.1, we have the following loc
    $ wc -l *SPECTUB*x
  1346 PinholeSPECTUB_Tools.cxx
  1113 PinholeSPECTUB_Weight3d.cxx
  1254 ProjMatrixByBinPinholeSPECTUB.cxx
   879 ProjMatrixByBinSPECTUB.cxx
  1055 SPECTUB_Tools.cxx
   989 SPECTUB_Weight3d.cxx

Roughly leading to
| Carles Falcon          |      3500
| Berta Marti-Fuster     |      1000
    
Summary for corrections to output:
| Author                 |    loc
| Carles Falcon          |   3500
| Berta Marti-Fuster     |   1000
| Claire Labbe           |   1000
| Mustapha Sadki         |    400
| Damiano Belluzzo       |     60
| Darren Hague           |     50
