#! /bin/sh
# git-fame allows printing "loc" (lines of code), "coms" (number of commits) and "fils" (number of files)
# per contributor (using .mailmap)
#
# Suggested usage of git fame:
#-w, --ignore-whitespace  Ignore whitespace when comparing the parent's
#                           version and the child's to find where the lines
#                           came from [default: False].
#  -M  Detect intra-file line moves and copies [default: False].
#  -C  Detect inter-file line moves and copies [default: False].
#
# These slow it down dramatically, but a lot of this has happened in STIR
git fame -w -M -C