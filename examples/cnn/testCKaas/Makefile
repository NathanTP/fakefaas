CC := gcc
CFLAGS := -O3 -g -Wall -Wextra -Werror -std=gnu99 
LIBS:=-L../libkaascnn/ -lkaascnn -lpthread
INCLUDES:=-I../libkaascnn

test: main.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) 

%.o: %.c
	$(CC) -c $< $(CFLAGS) $(INCLUDES)

clean:
	rm -f *.o test 
