CXX_FLAGS = -lzcm `pkg-config --libs --cflags opencv4`

cfg = ${PWD}/cfg/113/head_1

TARGET = tracker

VERSION = `git describe --tags`

all: build
# 	./${TARGET} -c SL_config.yaml -p pid
	./${TARGET} -c SL_Ltrains_config.yaml -p pid

rebuild: clean build

build:
	mkdir -p obj/
	g++ -O3 ${build_mode} -std=c++11 include/zcm_types/*pp include/vtracker/source/*.cpp include/utils/*pp include/*.cpp ${CXX_FLAGS} -c
	mv *.o obj/
	g++ -O3 ${build_mode} -std=c++11 main.cpp obj/*.o ${CXX_FLAGS} -o ${TARGET}
	find -name "*.gch" -exec rm -rf {} +

run:
# 	./${TARGET} -c SL_config.yaml -p pid
	./${TARGET} -c SL_Ltrains_config.yaml -p pid
# 	./${TARGET} --config=${cfg}/LL_config.yaml

clean:
	rm -rf obj/
	rm -f ${TARGET}
