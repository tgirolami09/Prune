#include <string>
#include <zip.h>
#include <zipconf.h>
using namespace std;
const int dataSize = 12*64*2+2*4;
zip_t** zip;
extern "C" {
void allocate_zips(int nbZips){
    zip = new zip_t*[nbZips];
}

void open_zip(char* filename, int id){
    int error;
    zip[id] = zip_open(filename, ZIP_FL_OVERWRITE|ZIP_FL_ENC_UTF_8, &error);
}

void add_data(int idZip, int id, char* data){
    zip_source_t* source = zip_source_buffer(zip[idZip], data, dataSize, 0);
    zip_file_add(zip[idZip], to_string(id).c_str(), source, ZIP_FL_OVERWRITE|ZIP_FL_COMPRESSED);
}

void read_data(int idZip, char* buffer, int id){
    zip_uint64_t index = zip_name_locate(zip[idZip], to_string(id).c_str(), 0);
    zip_file_t* zip_file = zip_fopen_index(zip[idZip], index, 0);
    zip_uint64_t num_read = zip_fread(zip_file, buffer, dataSize);
}
int get_nb_files(int idZip){
    return zip_get_num_entries(zip[idZip], 0);
}
}