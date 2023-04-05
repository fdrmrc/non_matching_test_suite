#!/bin/bash

if [ "$1" == "-h" ]; then 
        echo Usage: $0 [debug]
        echo
        echo Will run all tests in release mode, in serial.
        echo If [debug] is specified, then the programs are run in debug mode.
        echo The output is saved in the current directory.
        exit 0
fi

POSTFIX=""

if [ "$1" == "debug" ]; then 
    POSTFIX=".g"
fi

for prm in parameters/*prm; do
    exe=$(echo "$(basename $prm)" | cut -d'.' -f1)
    ./scripts/non_matching_test_suite.sh $exe$POSTFIX $prm
done