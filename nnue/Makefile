all: zipmanager.so

zipmanager.so: zipmanager.cpp
	$(CC) -fPIC $^ -O3 -shared -o $@ -lzip