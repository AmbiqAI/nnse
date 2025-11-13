# Generic Makefile for C project using ns-nnsp library


CC = gcc
CFLAGS = -I./ns-nnsp/includes-api -Wall -O2 -std=c99 -DFIXED_POINT=32
LDFLAGS = -L./ns-nnsp -lgeneric -lm


SRC = main.c def_nn3_se.c AudioPipe_wrapper.c input_data.c
OBJ = $(addprefix bin/, $(notdir $(SRC:.c=.o)))
TARGET = main


all: bin $(TARGET)


$(TARGET): $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) -o $@


bin/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@


bin:
	mkdir -p bin

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
