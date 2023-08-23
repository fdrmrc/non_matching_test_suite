#!/bin/bash
function run {
    docker run -ti -v `pwd`:`pwd` -u $(id -u):$(id -g) ghcr.io/fdrmrc/non_matching_test_suite:latest /bin/sh -c "$1"
}

if [ -z "$1" ]; then 
	echo Usage: $0 \[-np N\] program-name program-options
	echo
	echo Will run program-name with program-options, possibly via mpirun, passing -np N to mpirun.
	echo Here is a list of programs you can run:
	run "ls /usr/local/bin"
	echo 
	echo Programs ending with "".g"" are compiled with debug symbols.
	exit 0
fi

if [ "$1" == "-np" ]; then 
	run "cd `pwd`; mpirun -np $2 $3 ${@:4}"
else
	run "cd `pwd`; $1 ${@:2}"
fi
