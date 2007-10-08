#! /bin/sh
# a helper function to print the value of a single dicom field from a file
# args: filename field
# relies on dcmdump, a utility from dcmtk

if [ $# != 2 ]; then
  echo "Usage: ${0##*/} filename field" 1>&2
  exit 1
fi

  # use awk to print only 3rd column
  # use tr to get rid of square brackets in output of strings
  # next line failed for fields which contain spaces
  #dcmdump --load-short +P $2 $1 |awk '{print $3}'|tr -d '[]'
dcmdump --load-short +P $2 $1 |awk '{
  for (i=3; i<NF-3; i++)
  {  if ( $i ~ /#/ )
      break
    printf "%s ", $i
  }
  printf "\n"
}'|tr -d '[]'
  # try to test on status, but doesn't catch non-dicom file
  if [ ! $? ]; then
    echo "dcmdump failed on $2 $1" 1>&2
    exit 1
  fi

