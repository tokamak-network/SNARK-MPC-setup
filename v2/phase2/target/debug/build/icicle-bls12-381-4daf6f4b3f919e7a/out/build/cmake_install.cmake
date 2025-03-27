# Install script for directory: /home/mabingol/.cargo/git/checkouts/icicle-e4fed18c5a8319f4/9d1ad1c/icicle

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_device.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_device.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_device.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_device.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib" TYPE SHARED_LIBRARY FILES "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/build/icicle-bls12-381-4daf6f4b3f919e7a/out/build/libicicle_device.so")
  if(EXISTS "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_device.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_device.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_device.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_field_bls12_381.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_field_bls12_381.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_field_bls12_381.so"
         RPATH "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_field_bls12_381.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib" TYPE SHARED_LIBRARY FILES "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/build/icicle-bls12-381-4daf6f4b3f919e7a/out/build/libicicle_field_bls12_381.so")
  if(EXISTS "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_field_bls12_381.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_field_bls12_381.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_field_bls12_381.so"
         OLD_RPATH "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/build/icicle-bls12-381-4daf6f4b3f919e7a/out/build:"
         NEW_RPATH "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_field_bls12_381.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_curve_bls12_381.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_curve_bls12_381.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_curve_bls12_381.so"
         RPATH "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_curve_bls12_381.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib" TYPE SHARED_LIBRARY FILES "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/build/icicle-bls12-381-4daf6f4b3f919e7a/out/build/libicicle_curve_bls12_381.so")
  if(EXISTS "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_curve_bls12_381.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_curve_bls12_381.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_curve_bls12_381.so"
         OLD_RPATH "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/build/icicle-bls12-381-4daf6f4b3f919e7a/out/build:"
         NEW_RPATH "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/deps/icicle/lib/libicicle_curve_bls12_381.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/build/icicle-bls12-381-4daf6f4b3f919e7a/out/build/backend/cpu/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/mnt/d/MPC_ceremony/SNARK-MPC-setup/v2/phase2/target/debug/build/icicle-bls12-381-4daf6f4b3f919e7a/out/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
