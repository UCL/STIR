#! /bin/bash
set -e
org=${1/,v}
newloc=${2/,v}
if [ ${org} = ${newloc} ]; then
  # args are equal. skip
  exit 0
fi
org=`echo $org|sed -e s#Attic/##g` 
newloc=`echo $newloc|sed -e s#Attic/##g` 
cd ~/devel/STIR-hg/STIR
deadrev=`cvs log $org|head -n5|grep head|awk '{print $2}'`
if [ -z $deadrev ]; then
   echo 'no revision found. File $org exists?'
   exit 1
fi
echo "$org=>$newloc removing revs up to $deadrev"
# find any tags and remove from new
cvs log -h $org|awk -F':' -v newloc=$newloc '/\t.*: [0-9]\..*/ {tag=substr($1,2); cmd=sprintf("cvs tag -d %s %s", tag, newloc);system( cmd)}' 
cvs admin -o::${deadrev} $newloc
