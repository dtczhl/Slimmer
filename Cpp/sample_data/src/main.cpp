#include <iostream>
#include <fstream>

#include "grid.hpp"
#include "hierarchy.hpp"

int main(int argc, char*argv[])
{
    std::string sampler(argv[1]);

    if (sampler == "grid") {
        grid(argc, argv);
    } else if (sampler == "hierarchy") {
        hierarchy(argc, argv);
    }
    
    return 0;
}