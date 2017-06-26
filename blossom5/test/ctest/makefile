
test: myclib_test.cxx
	g++ -O0 -I../.. -L../.. -o exec myclib_test.cxx -lblossom
	LD_PRELOAD=../../libblossom.so ./exec

clean:
	rm -rf exec *~

