all:
	nvcc -std=c++14 -O2 -lcuda -lcudart -lcublas main.cpp -o main.out 
clean:
	rm -f main.out
run: all
	./main.out