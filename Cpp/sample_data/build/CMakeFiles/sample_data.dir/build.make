# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/build

# Include any dependencies generated for this target.
include CMakeFiles/sample_data.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sample_data.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sample_data.dir/flags.make

CMakeFiles/sample_data.dir/src/main.cpp.o: CMakeFiles/sample_data.dir/flags.make
CMakeFiles/sample_data.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sample_data.dir/src/main.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sample_data.dir/src/main.cpp.o -c /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/src/main.cpp

CMakeFiles/sample_data.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sample_data.dir/src/main.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/src/main.cpp > CMakeFiles/sample_data.dir/src/main.cpp.i

CMakeFiles/sample_data.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sample_data.dir/src/main.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/src/main.cpp -o CMakeFiles/sample_data.dir/src/main.cpp.s

CMakeFiles/sample_data.dir/src/grid.cpp.o: CMakeFiles/sample_data.dir/flags.make
CMakeFiles/sample_data.dir/src/grid.cpp.o: ../src/grid.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sample_data.dir/src/grid.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sample_data.dir/src/grid.cpp.o -c /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/src/grid.cpp

CMakeFiles/sample_data.dir/src/grid.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sample_data.dir/src/grid.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/src/grid.cpp > CMakeFiles/sample_data.dir/src/grid.cpp.i

CMakeFiles/sample_data.dir/src/grid.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sample_data.dir/src/grid.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/src/grid.cpp -o CMakeFiles/sample_data.dir/src/grid.cpp.s

# Object files for target sample_data
sample_data_OBJECTS = \
"CMakeFiles/sample_data.dir/src/main.cpp.o" \
"CMakeFiles/sample_data.dir/src/grid.cpp.o"

# External object files for target sample_data
sample_data_EXTERNAL_OBJECTS =

sample_data: CMakeFiles/sample_data.dir/src/main.cpp.o
sample_data: CMakeFiles/sample_data.dir/src/grid.cpp.o
sample_data: CMakeFiles/sample_data.dir/build.make
sample_data: CMakeFiles/sample_data.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable sample_data"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sample_data.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sample_data.dir/build: sample_data

.PHONY : CMakeFiles/sample_data.dir/build

CMakeFiles/sample_data.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sample_data.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sample_data.dir/clean

CMakeFiles/sample_data.dir/depend:
	cd /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/build /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/build /home/dtc/MyGit/dtc-sparseconvnet/Cpp/sample_data/build/CMakeFiles/sample_data.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sample_data.dir/depend

