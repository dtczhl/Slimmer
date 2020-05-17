#include <iostream>
#include <fstream>

#include <unistd.h>

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

    // plot memory
    std::string process_name;
    process_name = "/proc/self/statm";
    std::ifstream buffer(process_name.c_str());
    int tSize = 0, resident = 0, share = 0;
    buffer >> tSize >> resident >> share;
    buffer.close();
    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    std::cout << "Memory (MB) " << resident * page_size_kb / 1000  << std::endl;
    
    return 0;
}