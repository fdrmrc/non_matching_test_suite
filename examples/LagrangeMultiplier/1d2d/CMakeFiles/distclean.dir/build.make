# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/local/bin/cmake

# The command to remove a file.
RM = /opt/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/marcofeder/Desktop/Paper_NonMatching/Non_matching_test_suite/LagrangeMultiplier/1d2d

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/marcofeder/Desktop/Paper_NonMatching/Non_matching_test_suite/LagrangeMultiplier/1d2d

# Utility rule file for distclean.

# Include any custom commands dependencies for this target.
include CMakeFiles/distclean.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/distclean.dir/progress.make

CMakeFiles/distclean:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/marcofeder/Desktop/Paper_NonMatching/Non_matching_test_suite/LagrangeMultiplier/1d2d/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "distclean invoked"
	/opt/local/bin/cmake --build /Users/marcofeder/Desktop/Paper_NonMatching/Non_matching_test_suite/LagrangeMultiplier/1d2d --target clean
	/opt/local/bin/cmake --build /Users/marcofeder/Desktop/Paper_NonMatching/Non_matching_test_suite/LagrangeMultiplier/1d2d --target runclean
	/opt/local/bin/cmake -E remove_directory CMakeFiles
	/opt/local/bin/cmake -E remove CMakeCache.txt cmake_install.cmake Makefile build.ninja rules.ninja .ninja_deps .ninja_log

distclean: CMakeFiles/distclean
distclean: CMakeFiles/distclean.dir/build.make
.PHONY : distclean

# Rule to build all files generated by this target.
CMakeFiles/distclean.dir/build: distclean
.PHONY : CMakeFiles/distclean.dir/build

CMakeFiles/distclean.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/distclean.dir/cmake_clean.cmake
.PHONY : CMakeFiles/distclean.dir/clean

CMakeFiles/distclean.dir/depend:
	cd /Users/marcofeder/Desktop/Paper_NonMatching/Non_matching_test_suite/LagrangeMultiplier/1d2d && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/marcofeder/Desktop/Paper_NonMatching/Non_matching_test_suite/LagrangeMultiplier/1d2d /Users/marcofeder/Desktop/Paper_NonMatching/Non_matching_test_suite/LagrangeMultiplier/1d2d /Users/marcofeder/Desktop/Paper_NonMatching/Non_matching_test_suite/LagrangeMultiplier/1d2d /Users/marcofeder/Desktop/Paper_NonMatching/Non_matching_test_suite/LagrangeMultiplier/1d2d /Users/marcofeder/Desktop/Paper_NonMatching/Non_matching_test_suite/LagrangeMultiplier/1d2d/CMakeFiles/distclean.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/distclean.dir/depend
