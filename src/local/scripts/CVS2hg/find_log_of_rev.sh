#!/bin/sh
file=$1
rev=$2
#cvs log writes a lot of stuff before the log message and a line of "==============" at the end
# so, we use sed to look for the range between the line starting with "date:" and ending with "======"
# we only print the lines in between using a trick from http://www.grymoire.com/Unix/Sed.html#uh-35a
# we also need to skip any "branches" lines
cvs log -N  -r${rev} ${file}| sed -n '
 /^date:.*;/,/===============/ {
   /^date:.*;/n
   /^branches:/n
   /===============/ ! p
}
'