#! /bin/sh
# git-fame allows printing "loc" (lines of code), "coms" (number of commits) and "fils" (number of files)
# per contributor (using .mailmap)
#
# Suggested usage of git fame:
#  -w, --ignore-whitespace  Ignore whitespace when comparing the parent's
#                           version and the child's to find where the lines
#                           came from [default: False].
#  -M  Detect intra-file line moves and copies [default: False].
#  -C  Detect inter-file line moves and copies [default: False].
#
# In particular, `-C` takes ~30x longer to run, but a lot of this has happened in STIR
git fame -wMC --format json --excl '\.(eps|root|ahv|hv|v|hs|s|scan|l|hdr|rtf|gz|if|pdf|safir|options|png|cls|sty)$|external_helpers|crystal_map_front.txt|Doxyfile' \
  | tee git-fame-output.txt \
  | jq -c '{creators: [.data[] | {name: .[0]}]}' \
  | sed -r -e 's/(\{"name")/\n    \1/g' -e 's/:/: /g'
