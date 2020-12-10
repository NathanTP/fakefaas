CC := g++
CFLAGS := -O0 -g -Wall -Wextra -std=c++14 
# CFLAGS := -O0 -g -Wall -Wextra -Werror -std=c++14
LIBS:=-L../libcnn/ -lcnn -lpthread
INCLUDES:=-I../libcnn

test: main.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) 

%.o: %.cpp
	$(CC) -c $< $(CFLAGS) $(INCLUDES)

clean:
	rm -f *.o test 
