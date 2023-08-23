#!/bin/bash

cd ../build/examples/
export PARAM_DIR=../../data/
export BINARY_DIR=../../code/build/examples/

#Runs everything in Release mode.

#Tables 1,2,3
printf "Table 1\n"
./$BINARY_DIR/lm_1d2d_disk $PARAM_DIR/lm_1d2d_disk.smooth.prm
printf "Table 2\n"
./$BINARY_DIR/nitsche_1d2d_smooth_disk
printf "Table 3\n"
./$BINARY_DIR/cutfem_1d2d_smooth_disk



#Tables 4,5,6
printf "Table 4\n"
./$BINARY_DIR//lm_1d2d_flower $PARAM_DIR/lm_1d2d_flower.smooth.prm
printf "Table 5\n"
./$BINARY_DIR//nitsche_1d2d_smooth_flower
printf "Table 6\n"
./$BINARY_DIR//cutfem_1d2d_smooth_flower



#Tables 7,8,9
printf "Table 7\n"
./$BINARY_DIR/lm_1d2d_disk $PARAM_DIR/lm_1d2d_disk.non_smooth.prm
printf "Table 8\n"
./$BINARY_DIR/nitsche_1d2d_non_smooth_disk
printf "Table 9\n"
./$BINARY_DIR/cutfem_1d2d_non_smooth_disk



#Tables 10,11,12
printf "Table 10\n"
./$BINARY_DIR/lm_2d3d $PARAM_DIR/lm_2d3d.smooth_sphere.prm
printf "Table 11\n"
./$BINARY_DIR/nitsche_2d23_smooth_disk
printf "Table 12\n"
./$BINARY_DIR/cutfem_2d_3d_smooth_sphere



#Tables 13,14,15
printf "Table 13\n"
./$BINARY_DIR/lm_2d3d $PARAM_DIR/lm_2d3d.non_smooth_sphere.prm
printf "Table 14\n"
./$BINARY_DIR/nitsche_2d23_non_smooth_disk
printf "Table 15\n"
./$BINARY_DIR/cutfem_2d_3d_non_smooth_sphere