#! /bin/sh
# sed script changes "Copyright ... $Date: 2014-02-15 00:17:41 $" to "Copyright ... 2009 ..."
# and remove lines with $Id: remove_cvs_keywords.sh,v 1.1 2014-02-15 00:17:41 kris Exp $ and only $date$ and $Revision: 1.1 $

file=$1
TMPFILE=/tmp/remove_cvs_keywords.tmp

sed -e's/\(Copyright.*\)\$Date: 2014-02-15 00:17:41 $/\1\2/' -e '/[/#\s]*\$Id: remove_cvs_keywords.sh,v 1.1 2014-02-15 00:17:41 kris Exp $\s*$/d' -e'/[/#\s]*$Date: 2014-02-15 00:17:41 $\s*$/d' -e'/[/#\s]*$Revision: 1.1 $\s*$/d' $file > $TMPFILE

if cmp -s $file $TMPFILE
then
  # identical, so don't replace
  rm $TMPFILE
else
  rm $file
  mv $TMPFILE $file
fi
