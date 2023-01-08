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
# Running from 2003 (rel_1_10) as various files were apparently not in git before that (they were in CVS)
# The file therefore needs manual ordering (see below)
git fame -wMC --since=2003-06-27 --excl '\.(eps|root|ahv|hv|v|hs|s|scan|l|hdr|rtf|gz|if|pdf|safir|options|png|cls|sty)$|external_helpers|crystal_map|collimator.*txt|Doxyfile.in|LICENSE.txt|LICENSES' \
  | tee git-fame-output.txt
exit

# git history doesn't tell us how many slocs the PARAPET people had
# KT has ordered them roughly in order of sloc
# and suggests to insert them before the "trivial" contributors (e.g. after ~60 sloc)
    {"name": "Zverovich, Alexey"},
    {"name": "Jacobson, Matthew"},
    {"name": "Labbe, Claire"},
    {"name": "Sadki, Mustapha"},
    {"name": "Hague, Darren"},
    {"name": "Belluzzo, Damiano"},
    {"name": "Valente, Patrick"},
