CXX = g++
CXXFLAGS += -W -Wall -pedantic -std=c++98 -g -O3 -fPIC

all: nonlinear_.so linalg_.so _topic_modeling.so

nonlinear_.so: nonlinear_.c
	python setup.py install --install-platlib=.

linalg_.so: linalg_.c
	python setup.py install --install-platlib=.

_topic_modeling.so: _topic_modeling.o
	${CXX} ${CXXFLAGS} -shared -o $@ $^

_topic_modeling.o: _topic_modeling.c
	${CXX} ${CXXFLAGS} -c -o $@ $^

clean:
	rm -r build
	rm linalg_.so
	rm nonlinear_.so
	rm _topic_modeling.so
	rm _topic_modeling.o
