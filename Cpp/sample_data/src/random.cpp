#include "random.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/random_simplify_point_set.h>
#include <CGAL/property_map.h>
#include <CGAL/IO/read_xyz_points.h>

#include <vector>
#include <fstream>
#include <iostream>

#include <cmath> 

#include <boost/tuple/tuple.hpp>

// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
typedef boost::tuple<Point, float, float, float, float> MYPoint;

void random(int argc, char*argv[]) {
/*
Input:
    argv[0]: this program
    argv[1]: "random"
    argv[2]: dir of output data
    argv[3]: filename of input data under ../tmp/
    argv[4]: keep ratio in pecentage [0, 100]
*/

    if (argc != 5) {
        std::cerr << "[grid.cpp] require argc: 5, input argc: " << argc << std::endl;
        std::cerr << "Format: program random output_dir input_file keep_ratio" << std::endl;
        exit(-1);
    }

    std::string sampler(argv[1]);
    std::string dstDir(argv[2]);
    std::string srcFile(argv[3]);
    double keep_ratio = atof(argv[4]);
    
    // std::cout << "sampler: " << sampler << ", dstDir: " << dstDir << ", srcFile: " << srcFile << ", keep ratio: " << keep_ratio << std::endl;

    std::vector<MYPoint> points;
    std::string srcFilename = "../tmp/" + srcFile;

    // read points 
    // std::cout << "Reading file: " << srcFilename << std::endl;
    std::FILE *pFile = fopen(srcFilename.c_str(), "rb");
    fseek(pFile, 0, SEEK_END);    // file pointer goes to the end of the file
    long fileSize = ftell(pFile); // file size
    rewind(pFile);                // rewind file pointer to the beginning
    float *rawData = new float[fileSize];
    fread(rawData, sizeof(float), fileSize/sizeof(float), pFile);
    int nProperties = 7;
    long number_of_points = fileSize / nProperties / sizeof(float); // x, y, z, r, g, b, label

    for (int i = 0; i < number_of_points; i++) {
        points.push_back(MYPoint(Point(rawData[nProperties*i], rawData[nProperties*i+1], rawData[nProperties*i+2]), 
            rawData[nProperties*i+3], rawData[nProperties*i+4], rawData[nProperties*i+5], rawData[nProperties*i+6]));
    }

    // std::cout << "Point number of points: " << points.size() << std::endl;

    /*
    std::cout << "Before data sampling" << std::endl;
    for (int i = 0; i < 10; i++) {   
        float x = points[i].get<0>().x();
        float y = points[i].get<0>().y();
        float z = points[i].get<0>().z();
        float r = points[i].get<1>();
        float g = points[i].get<2>();
        float b = points[i].get<3>();
        float label = points[i].get<4>();
        std::cout << "x:" << x << " y: " << y << " z: " << z << " r: " << r << " g: " << g << " b: " << b << " l: " << label << std::endl; 
    }
    */

    // std::cout << "From cpp keep ratio " << 100 - keep_ratio << std::endl; 
    points.erase(CGAL::random_simplify_point_set(points.begin(), points.end(), CGAL::Nth_of_tuple_property_map<0, MYPoint>(), 100.0-keep_ratio),
        points.end());
    
    /*
    std::cout << "After data sampling" << std::endl;
    for (int i = 0; i < 10; i++) {   
        float x = points[i].get<0>().x();
        float y = points[i].get<0>().y();
        float z = points[i].get<0>().z();
        float r = points[i].get<1>();
        float g = points[i].get<2>();
        float b = points[i].get<3>();
        float label = points[i].get<4>();
        std::cout << "x:" << x << " y: " << y << " z: " << z << " r: " << r << " g: " << g << " b: " << b << " l: " << label << std::endl; 
    }
    */

    std::string dstFileSave = srcFilename + ".trim";
    // std::cout << dstFileSave << std::endl;

    // std::cout << "Original number of points: " << number_of_points << ", After trim: " << points.size() << std::endl;

    std::ofstream out(dstFileSave, std::ios_base::binary);
    for (int i = 0; i < points.size(); i++) {
            
        float x = points[i].get<0>().x();
        float y = points[i].get<0>().y();
        float z = points[i].get<0>().z();
        float r = points[i].get<1>();
        float g = points[i].get<2>();
        float b = points[i].get<3>();
        float label = points[i].get<4>();

        // std::cout << x << " " << y << " " << z << " " << r << " " << g << " " << b << " " << label << std::endl; 
        
        out.write((char *)&x, sizeof(float));
        out.write((char *)&y, sizeof(float));
        out.write((char *)&z, sizeof(float));
        out.write((char *)&r, sizeof(float));
        out.write((char *)&g, sizeof(float));
        out.write((char *)&b, sizeof(float));
        out.write((char *)&label, sizeof(float));
    }
    out.close();
}