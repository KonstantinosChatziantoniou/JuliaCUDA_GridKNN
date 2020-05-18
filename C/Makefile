CC = nvcc
CFLAGS = -O3
CUFLAGS = -O3 -lcuda -lcudart # -Xptxas -v#,-abi=no #Will print the number of lmembytes for each kernel (only if kernel uses LMEM)
#-std=c99 --default-stream per-thread
LIBS = -lm #-Wall -Wextra
OBJDIR = ./bin
HDRDIR = ./headers
SRCDIR = ./src

_OBJ =  main.o preprocess.o helpers.o kernel_simple.o kernel_with_skip.o
OBJ = $(patsubst %, $(OBJDIR)/%, $(_OBJ))

_DEPS = preprocess.h helpers.h kernel_with_skip.h kernel_simple.h
DEPS = $(patsubst %, $(HDRDIR)/%, $(_DEPS))


mainProgram: $(OBJ)
	$(CC) -o $@ $^ $(CUFLAGS) $(LIBS)


$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CC) -c -o $@ $<  $(CUFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	gcc -Ofast -c -o $@ $<  $(CFLAGS)

clean:
	rm -rf ./*.csv; rm -rf ./bin/*.o; rm -rf mainProgram

clear: 
	rm -rf ./*.csv; 