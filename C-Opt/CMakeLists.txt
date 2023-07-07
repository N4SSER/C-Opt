cmake_minimum_required(VERSION 3.25)
project(C_Opt)

set(CMAKE_CXX_STANDARD 17)

add_library(C_Opt src/algorithms/moo/GA.cpp src/algorithms/moo/GA.h src/algorithms/moo/NSGA2.cu src/algorithms/moo/NSGA2.cuh)
