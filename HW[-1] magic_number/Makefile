# b04902053 鄭淵仁 #

all: magic.o
	gcc magic.o -o magic
%.o: %.cc
	g++ $< -c

.PHONY: run, clean
run:
	./magic
clean:
	@rm -rf *.o
