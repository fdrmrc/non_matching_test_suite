#!/bin/bash

cd ../build/examples/
export PARAM_DIR=../../data/
export BINARY_DIR=../../code/build/examples/
export BINARY_POSTFIX=""

#Runs everything in Release mode.

#Tables 1,2,3
echo Table 1
./$BINARY_DIR/lm_1d2d_disk$BINARY_POSTFIX $PARAM_DIR/lm_1d2d_disk.smooth.prm
echo Table 2
./$BINARY_DIR/nitsche_1d2d_smooth_disk$BINARY_POSTFIX
echo Table 3
./$BINARY_DIR/cutfem_1d2d_smooth_disk$BINARY_POSTFIX

#Tables 4,5,6
echo Table 4
./$BINARY_DIR/lm_1d2d_flower$BINARY_POSTFIX $PARAM_DIR/lm_1d2d_flower.smooth.prm
echo Table 5
./$BINARY_DIR/nitsche_1d2d_smooth_flower$BINARY_POSTFIX
echo Table 6
./$BINARY_DIR/cutfem_1d2d_smooth_flower$BINARY_POSTFIX

#Tables 7,8,9
echo Table 7
./$BINARY_DIR/lm_1d2d_disk$BINARY_POSTFIX $PARAM_DIR/lm_1d2d_disk.non_smooth.prm
echo Table 8
./$BINARY_DIR/nitsche_1d2d_non_smooth_disk$BINARY_POSTFIX
echo Table 9
./$BINARY_DIR/cutfem_1d2d_non_smooth_disk$BINARY_POSTFIX

#Tables 10,11,12
echo Table 10
./$BINARY_DIR/lm_2d3d$BINARY_POSTFIX $PARAM_DIR/lm_2d3d.smooth_sphere.prm
echo Table 11
./$BINARY_DIR/nitsche_2d23_smooth_disk$BINARY_POSTFIX
echo Table 12
./$BINARY_DIR/cutfem_2d_3d_smooth_sphere$BINARY_POSTFIX

#Tables 13,14,15
echo Table 13
./$BINARY_DIR/lm_2d3d$BINARY_POSTFIX $PARAM_DIR/lm_2d3d.non_smooth_sphere.prm
echo Table 14
./$BINARY_DIR/nitsche_2d23_non_smooth_disk$BINARY_POSTFIX
echo Table 15
./$BINARY_DIR/cutfem_2d_3d_non_smooth_sphere$BINARY_POSTFIX
