#include <iostream>
#include <fstream>

#include "grid.hpp"
#include "hierarchy.hpp"
#include "random.hpp"

int main(int argc, char*argv[])
{
    std::string sampler(argv[1]);

    if (sampler == "grid") {
        grid(argc, argv);
    } else if (sampler == "hierarchy") {
        hierarchy(argc, argv);
    } else if (sampler == "random") {
        random(argc, argv);
    } else {
        std::cerr << "Unknown sampler: " << sampler << std::endl;
        exit(-1);
    }
    
    return 0;
}