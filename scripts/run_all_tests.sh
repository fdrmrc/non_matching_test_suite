#!/bin/bash

cd ../build/examples/
export PARAM_DIR=../../parameters/

#Tables 1,2,3
./lm_1d2d_disk $PARAM_DIR/lm_1d2d_disk.smooth.prm
./nitsche_1d2d_smooth_disk
./cutfem_1d2d_smooth_disk



#Tables 4,5,6
./lm_1d2d_flower $PARAM_DIR/lm_1d2d_flower.smooth.prm
./nitsche_1d2d_smooth_flower
./cutfem_1d2d_smooth_flower



#Tables 7,8,9
./lm_1d2d_disk $PARAM_DIR/lm_1d2d_disk.non_smooth.prm
./nitsche_1d2d_non_smooth_disk
./cutfem_1d2d_non_smooth_disk



#Tables 10,11,12
./lm_2d3d $PARAM_DIR/lm_2d3d.smooth_sphere.prm
./nitsche_2d23_smooth_disk
./cutfem_2d_3d_smooth_sphere



#Tables 13,14,15
./lm_2d3d $PARAM_DIR/lm_2d3d.non_smooth_sphere.prm
./nitsche_2d23_non_smooth_disk
./cutfem_2d_3d_non_smooth_sphere


# if [ "$1" == "-h" ]; then 
#         echo Usage: $0 [debug]
#         echo
#         echo Will run all tests in release mode, in serial.
#         echo If [debug] is specified, then the programs are run in debug mode.
#         echo The output is saved in the current directory.
#         exit 0
# fi

# POSTFIX=""

# if [ "$1" == "debug" ]; then 
#     POSTFIX=".g"
# fi

# for prm in parameters/*prm; do
#     exe=$(echo "$(basename $prm)" | cut -d'.' -f1)
#     ./scripts/non_matching_test_suite.sh $exe$POSTFIX $prm
# done