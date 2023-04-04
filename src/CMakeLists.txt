# src/CMakeLists.txt

# set up shared library by hand:
ADD_LIBRARY(Non_matching_test_suite SHARED coupling.cc non_matching_utilities.cc)
DEAL_II_SETUP_TARGET(Non_matching_test_suite)
