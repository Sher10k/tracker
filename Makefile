CXX_FLAGS = -lzcm `pkg-config --libs --cflags opencv4`

cfg = ${PWD}/config

TARGET = tracker

VERSION = `git describe --tags`

all: build
	./${TARGET} -c ${cfg}/SL_config.yaml -p ${PWD}/pid

rebuild: clean build

build:
	mkdir -p obj/
	g++ -O3 ${build_mode} -std=c++11 include/zcm_types/*pp include/vtracker/source/*.cpp include/utils/*pp include/*.cpp ${CXX_FLAGS} -c
	mv *.o obj/
	g++ -O3 ${build_mode} -std=c++11 main.cpp obj/*.o ${CXX_FLAGS} -o ${TARGET}
	find -name "*.gch" -exec rm -rf {} +

run:
	./${TARGET} -c ${cfg}/SL_config.yaml -p ${PWD}/pid

clean:
	rm -rf obj/
	rm -f ${TARGET}
