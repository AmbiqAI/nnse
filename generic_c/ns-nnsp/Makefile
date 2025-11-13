SRCS := $(wildcard src/*.c)
# If FIXED_POINT is set to 32, add kiss_fft.c and kiss_fftr.c explicitly
OBJS := $(patsubst src/%.c,bin/%.o,$(SRCS))
LIB := libnnsp.a

CC = arm-none-eabi-gcc
CFLAGS ?= -O3 -mthumb -mcpu=cortex-m55 -mfloat-abi=hard -ffunction-sections -fdata-sections -fomit-frame-pointer -fno-exceptions -std=c99

CFLAGS += -DFIXED_POINT=32 # Added for kiss_fft and kiss_fftr, specifically for int32

AR ?= arm-none-eabi-ar

all: bin $(LIB)

bin:
	mkdir -p bin

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

bin/%.o: src/%.c
	$(CC) $(CFLAGS) -I./includes-api -Isrc -I../../includes/extern/CMSIS/CMSIS-DSP-1.16.2/Include -I../../includes/extern/CMSIS/CMSIS_5-5.9.0/CMSIS/Core/Include -c $< -o $@

clean:
	rm -f bin/*.o $(LIB)
