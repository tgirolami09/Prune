#include <zip.h>
const int dataSize = 12*64*2+2*4;
zip_t** zip;
int numzips;
extern "C" {
void allocate_zips(int nbZips){
    zip = new zip_t*[nbZips];
    numzips = nbZips;
}

void open_zip(char* filename, int id){
    int error;
    zip[id] = zip_open(filename, ZIP_CREATE, &error);
}

void add_data(int idZip, int id, char* data){
    zip_source_t* source = zip_source_buffer(zip[idZip], data, dataSize, 0);
    char snum[5];
    sprintf(snum, "%4x", id);
    zip_file_add(zip[idZip], snum, source, ZIP_FL_OVERWRITE|ZIP_FL_COMPRESSED);
    zip_source_free(source);
}

void read_data(int idZip, char* buffer, int id){
    char snum[5];
    zip_uint64_t index = zip_name_locate(zip[idZip], snum, 0);
    zip_file_t* zip_file = zip_fopen_index(zip[idZip], index, 0);
    zip_uint64_t num_read = zip_fread(zip_file, buffer, dataSize);
    zip_fclose(zip_file);
}
int get_nb_files(int idZip){
    return zip_get_num_entries(zip[idZip], 0);
}

void clear_zip(int idZip){
    zip_close(zip[idZip]);
}

}