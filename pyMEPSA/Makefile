CC=gcc

all: mepsa.so

mepsa.so: mepsa.o
	$(CC) -shared -o mepsa.so mepsa.o

mepsa.o: mepsa.c
	$(CC) -c -Wall -Wno-error -fpic mepsa.c	

clean:
	rm -f mepsa.o mepsa.so
