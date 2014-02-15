#!/bin/sh
file=$1
rev=$2
sed -n "
 /^${rev}$/ {
  n
  p
  q
}
" $file