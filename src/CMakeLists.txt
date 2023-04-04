# src/CMakeLists.txt

# set up shared library by hand:
ADD_LIBRARY(Non_matching_test_suite SHARED coupling_utilities.cc intersections.cc)
DEAL_II_SETUP_TARGET(Non_matching_test_suite)
