find_package(NLopt REQUIRED)

###########
## Build ##
###########
set(LIB_NLOPT ifopt_nlopt)
# The IPOPT solver interface
add_library(${LIB_NLOPT} 
    src/nlopt_solver.cc
)
target_include_directories(${LIB_NLOPT} 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${LIB_NLOPT}
  PUBLIC 
    ${LIB_CORE}
  PRIVATE
    NLopt::nlopt
)


#############
## Testing ##
#############
add_executable(${LIB_NLOPT}-example test/ex_test_nlopt.cc)
target_link_libraries(${LIB_NLOPT}-example 
  PRIVATE 
    ${LIB_NLOPT}
)
add_test(${LIB_NLOPT}-example ${LIB_NLOPT}-example)


#############
## Install ##
#############
 # Copy library files to usr/local/lib/libifopt_nlopt.so
install(
  TARGETS ${LIB_NLOPT}
  EXPORT  ${LIB_NLOPT}-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
# install example executable
install(
  TARGETS ${LIB_NLOPT}-example
  RUNTIME DESTINATION ${IFOPT_INSTALL_BINDIR}
)
# Copy header files to usr/local/include/ifopt/*
install(DIRECTORY include/ifopt/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ifopt
)
# install a *.cmake file in destination
install(EXPORT ${LIB_NLOPT}-targets
  FILE ${LIB_NLOPT}-targets.cmake
  NAMESPACE ifopt::
  DESTINATION ${config_package_location}
)


