DIRS := . MinCost GEOM

SOURCES := $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
OBJS := $(patsubst %.cpp, %.o, $(SOURCES))

CFLAGS := -O3 -D_NDEBUG
CXX ?= g++
LIBS := -lm
INCLUDES :=
LIBDIR :=

# Add librt if the target platform is not Darwin (OS X)
ifneq ($(shell uname -s),Darwin)
    LIBS += -lrt
endif

all: clean libblossom

libblossom: ${OBJS}
	$(CXX) $(CFLAGS) -shared -Wl,-soname,libblossom ${LIBDIR} -o libblossom.so ${OBJS} ${LIBS}

.cpp.o:
	$(CXX) $(CFLAGS) -fPIC ${INCLUDES} $< -c -o $@

clean:
	rm -f ${OBJS} blossom5 *~
