
test: myclib_test.cxx
	g++ -O3 -I../.. -L/home/iashraf/Desktop/blossom5/blossom5lib -o exec myclib_test.cxx -lblossom
	LD_PRELOAD=/home/iashraf/Desktop/blossom5/blossom5lib/libblossom.so ./exec

clean:
	rm -rf exec *~

