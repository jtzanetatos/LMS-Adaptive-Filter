#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#include <iostream>
#include <sstream>
#include <cmath>


std::string getcwd() {
        char buff[FILENAME_MAX];
        GetCurrentDir( buff, FILENAME_MAX );
        std::string current_working_dir(buff);
        return current_working_dir;
}

int main(int argc, char* argv[]) {

        // CommandLineParser parser(argc, argv, params);

        if (argc == 1) {

        }
        else {

        }
        // Initialize file pointers
        // FILE *input_signal, *out_signal, *error_rate;

        std::cout << "Current path is: " << getcwd() << std::endl;


}
