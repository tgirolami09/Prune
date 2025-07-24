#include <chrono>
#include <cstdio>
#include <string>
using namespace std;

int main(int argc, char** argv){
    string command1 = argv[1];
    string command2 = argv[2];
    command1 += " > /dev/null";
    command2 += " > /dev/null";
    const char* command1_str=command1.c_str();
    const char* command2_str=command2.c_str();
    int times = atoi(argv[3]);
    double t1=0, t2=0;
    int percent=times/100;
    int part=percent;
    for(int i=0; i<times; i++){
        if(i >= part){
            printf("\r%d%%", 100*i/times);
            fflush(stdout);
            part += percent;
        }
        const auto start1{std::chrono::steady_clock::now()};
        if(system(command1_str))break;;
        const auto finish1{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds1{finish1 - start1};
        t1 += elapsed_seconds1.count();

        const auto start2{std::chrono::steady_clock::now()};
        if(system(command2_str))break;;
        const auto finish2{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds2{finish2 - start2};
        t2 += elapsed_seconds2.count();
    }
    printf("\r100%%\n");
    double r=t2/t1;
    string word="faster";
    if(r < 1){
        r = 1/r;
        word = "slower";
    }
    if(r > 2){
        printf("the first command is %.2fx %s than the second command\n", r, word.c_str());
    }else{
        printf("the first command is %.2f%% %s than the second command\n", (r-1)*100, word.c_str());
    }
    printf("%.3f %.3f\n", t1, t2);
}