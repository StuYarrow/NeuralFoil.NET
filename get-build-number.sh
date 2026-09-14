#!/bin/bash

while getopts M:m: flag
do
    case "${flag}" in
        M) major=${OPTARG};;
        m) minor=${OPTARG};;
    esac
done

latestBuild=`git ls-remote -t -q \
	| grep -o "v${major}.${minor}.[[:digit:]]*$" \
	| sort -r --version-sort \
	| head -n 1 \
	| cut -d "." -f 3`

build=0

re='^[0-9]+$'
if [[ ${latestBuild} =~ $re ]] ; then
	build=$((${latestBuild} + 1))
fi

echo ${build}