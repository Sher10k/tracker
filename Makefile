CXX_FLAGS = -lzcm `pkg-config --libs --cflags opencv4`

cfg = ${PWD}/cfg/113/head_1

VERSION = `git describe --tags`

all: build
	./tracker -c SL_config.yaml -p pid
# 	./tracker -c SL_Ltrains_config.yaml -p pid

build:
	mkdir -p obj/
	g++ -O3 ${build_mode} -std=c++11 include/zcm_types/*pp include/vtracker/header/* include/utils/*pp include/*.cpp ${CXX_FLAGS} -c
	mv *.o obj/
	g++ -O3 ${build_mode} -std=c++11 main.cpp obj/*.o ${CXX_FLAGS} -o tracker
	find -name "*.gch" -exec rm -rf {} +

run:
	./tracker -c SL_config.yaml -p pid
# 	./tracker -c SL_Ltrains_config.yaml -p pid
# 	./tracker --config=${cfg}/LL_config.yaml
