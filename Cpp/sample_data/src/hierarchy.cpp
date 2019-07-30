#include "hierarchy.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/hierarchy_simplify_point_set.h>

#include <vector>
#include <fstream>
#include <iostream>

// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;

void hierarchy(int argc, char*argv[]) {

    if (argc != 7) {
        std::cerr << "argc: " << argc << " input argc = " << argc << std::endl;
    }

    std::string sampler(argv[1]);
    int start = atoi(argv[2]);
    int end = atoi(argv[3]);
    int step = atoi(argv[4]);
    std::string dstDir(argv[5]);
    std::string dstFile(argv[6]);

    /*
    std::cout << "sampler: " << sampler
        << ", start: " << start << ", end: " << end << ", step: " << step 
        << ", dstDir: " << dstDir << ", dstFile: " << dstFile << std::endl;
    */

    std::vector<Point> points;

    std::string srcFilename = "../tmp/" + dstFile;

    // read points 
    // std::cout << "Reading file: " << srcFilename << std::endl;
    std::FILE *pFile = fopen(srcFilename.c_str(), "rb");
    fseek(pFile, 0, SEEK_END);    // file pointer goes to the end of the file
    long fileSize = ftell(pFile); // file size
    rewind(pFile);                // rewind file pointer to the beginning
    float *rawData = new float[fileSize];
    fread(rawData, sizeof(float), fileSize/sizeof(float), pFile);
    long number_of_points = fileSize / 3 / sizeof(float); // x, y, z

    for (int i = 0; i < number_of_points; i++) {
        points.push_back(Point(rawData[3*i], rawData[3*i+1], rawData[3*i+2]));
    }

    std::cout << "Point number of points: " << points.size() << std::endl;
}