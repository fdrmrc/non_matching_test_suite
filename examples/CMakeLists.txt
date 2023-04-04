# set up executable with autopilot macro:
SET(TARGET "Test_LM")
SET(TARGET_SRC LagrangeMultiplier/2d3d/lagrange_multiplier.cc)
DEAL_II_INVOKE_AUTOPILOT()
TARGET_LINK_LIBRARIES(${TARGET} Non_matching_test_suite)