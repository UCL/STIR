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
deadrev=`cvs log -h $org|head -n5|grep head|awk '{print $2}'`
if [ -z "$deadrev" ]; then
   echo 'no revision found. File $org exists?'
   exit 1
fi
echo "$org=>$newloc removing revs up to $deadrev"
# find any tags in old and remove from new
cvs log -h $org|awk -F':' -v newloc=$newloc '/\t.*: [0-9]\..*/ {tag=substr($1,2); if (tag != OBJFUNCbranch) { cmd=sprintf("cvs tag -d %s %s", tag, newloc);system( cmd) } }' 
objfuncbranch=`cvs log -h $newloc | awk '/\tOBJFUNCbranch/ { print $2 }'`
if [ ! -z "${objfuncbranch}" ]; then
  # find out where the branch started
  objfuncfirstrev=`echo $objfuncbranch| awk -F. '{ print $1 "." $2 "." $4 ".1" }'`
  # also remove any OBJFUNC tags in new
  cvs log -h $newloc|awk -F':' -v newloc=$newloc '/\t.*: [0-9]\.[0-9]+\.[1-9].*/ {tag=substr($1,2); cmd=sprintf("cvs tag -d %s %s", tag, newloc);system( cmd)}' 
  # remove all these revisions
  if cvs admin -o${objfuncfirstrev}: $newloc
  then : 
  fi
  # we need to manually manipulate the RCS file to remove the whole branch
  rcsfile=`cvs log -R ${newloc}`
  sed -e 's/\tOBJFUNCbranch':${objfuncbranch}// ${rcsfile} > ${rcsfile}.new
  mv ${rcsfile}.new ${rcsfile}
fi
# finally
cvs admin -o::${deadrev} $newloc
