SOURCES := myclib.cpp
OBJS := $(patsubst %.cpp, %.o, $(SOURCES))

CFLAGS := -O3 -std=c++11
CXX := g++
LIBS := -lm
INCLUDES :=
LIBDIR :=

SONAME:= libmyclib

all: clean $(SONAME).so

$(SONAME).so: ${OBJS}
	$(CXX) $(CFLAGS) -shared -Wl,-soname,$(SONAME) ${LIBDIR} -o $(SONAME).so ${OBJS} ${LIBS}

.cpp.o:
	$(CXX) $(CFLAGS) -fPIC ${INCLUDES} $< -c -o $@

clean:
	rm -f ${OBJS} $(SONAME).so *~
