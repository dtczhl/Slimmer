#include <iostream>
#include <fstream>

#include "grid.hpp"

int main(int argc, char*argv[])
{
    std::string sampler(argv[1]);

    if (sampler == "grid") {
        grid(argc, argv);
    }
    
    return 0;
}