#! /bin/sh
if [ $# -ne 3 ]; then
    echo "usage: $0 prefix postfix numgates"
    exit 1
fi

filenames="{"
prefix=$1
postfix=$2
max=$3
for g in `count 1 $(($max-1))`; do
  filenames="${filenames}${prefix}$g${postfix},"
done
filenames="${filenames}${prefix}$max${postfix}}"

rm -f ${prefix}allgates${postfix}
cat <<EOF > ${prefix}allgates${postfix}
Multigate:=
filenames:=$filenames
end:=
EOF
