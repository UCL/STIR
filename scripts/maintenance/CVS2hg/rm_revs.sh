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
deadrev=`cvs log -h $org|head -n5|grep ^head:|awk '{print $2}'`
if [ -z "$deadrev" ]; then
   echo "no revision found. File $org exists?"
   exit 1
fi
# check if newloc file exists and is ok
if cvs log -S -R ${newloc} > /dev/null 2>&1
then
  :
else
   echo "file $newloc is not in CVS"
   exit 1
fi
# find previous rev
previousrev=`echo $deadrev|awk -F. '{ printf("%d.%d", $1,$2-1) }'`

# check if actually the same file
if [ xx"`~/devel/hgroot/find_log_of_rev.sh $org $previousrev`" != xx"`~/devel/hgroot/find_log_of_rev.sh $newloc $previousrev`" ]; then
  echo "log messages do not match at rev $previousrev. leave $newloc alone ($org)"
  exit 0
fi
cvs up -p -ko -r$previousrev $org > /tmp/cvs2hg.tmp1 2> /dev/null
cvs up -p -ko -r$previousrev $newloc > /tmp/cvs2hg.tmp2 2> /dev/null
if cmp /tmp/cvs2hg.tmp1 /tmp/cvs2hg.tmp2 > /dev/null
then
 :
else
  echo "content does not match at rev $previousrev. leave $newloc alone ($org)"
  exit 0
fi


echo "$org=>$newloc removing revs up to $previousrev"
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
# create "move" revision
oldrcsfile=`cvs log -R ${org}`
newrcsfile=`cvs log -R ${newloc}`
movedateline=`~/devel/hgroot/find_dateline_of_rev.sh ${oldrcsfile} ${deadrev}|sed -e 's/state dead/state Exp/'`
echo "Replacing date for ${previousrev} with \"${movedateline}\""
~/devel/hgroot/replace_dateline_of_rev.sh ${newrcsfile} ${previousrev} "${movedateline}"
# replace log message
logmessage=`~/devel/hgroot/find_log_of_rev.sh ${org} ${deadrev}`
cvs admin -m${previousrev}:"${logmessage}" ${newloc}
# finally
cvs admin -o::${previousrev} $newloc
