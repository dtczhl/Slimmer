#include "grid.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/grid_simplify_point_set.h>
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

void grid(int argc, char*argv[]) {

    if (argc != 7) {
        std::cerr << "argc: " << argc << " input argc = " << argc << std::endl;
    }

    std::string sampler(argv[1]);
    int start = atoi(argv[2]);
    int end = atoi(argv[3]);
    int step = atoi(argv[4]);
    std::string dstDir(argv[5]);
    std::string dstFile(argv[6]);

    
    std::cout << "sampler: " << sampler
        << ", start: " << start << ", end: " << end << ", step: " << step 
        << ", dstDir: " << dstDir << ", dstFile: " << dstFile << std::endl;

    std::vector<MYPoint> points;

    std::string srcFilename = "../tmp/" + dstFile;

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

    std::cout << "Point number of points: " << points.size() << std::endl;

/*
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

    std::vector<double> cell_size_vec;
    std::vector<int> cell_num_vec;
    double min_cell_size = 0.001, max_cell_size = 0.01;

    while (true) {
        std::vector<MYPoint> points_copy = points;
        points_copy.erase(CGAL::grid_simplify_point_set(points_copy.begin(), points_copy.end(), CGAL::Nth_of_tuple_property_map<0, MYPoint>(), max_cell_size),
            points_copy.end());
        if (1.0*points_copy.size()/number_of_points > 0.05) {
            max_cell_size *= 2.0;
        } else {
            break;
        }
    }
    while (true) {
        std::vector<MYPoint> points_copy = points;
        points_copy.erase(CGAL::grid_simplify_point_set(points_copy.begin(), points_copy.end(), CGAL::Nth_of_tuple_property_map<0, MYPoint>(), min_cell_size),
            points_copy.end());
        if (1.0*points_copy.size()/number_of_points < 0.95) {
            min_cell_size /= 2.0;
        } else {
            break;
        }
    }

    int nStep = 1000;
    for (int i = 0; i < nStep; i++) {
        cell_size_vec.push_back((max_cell_size - min_cell_size)/nStep*i + min_cell_size);

        std::vector<MYPoint> points_copy = points;
        points_copy.erase(CGAL::grid_simplify_point_set(points_copy.begin(), points_copy.end(), CGAL::Nth_of_tuple_property_map<0, MYPoint>(), cell_size_vec[i]),
            points_copy.end());

        cell_num_vec.push_back(points_copy.size());
    }
/*
    for(int i = 0; i < cell_num_vec.size(); i++) {
        std::cout << cell_num_vec[i] << " " << 1.0*cell_num_vec[i]/number_of_points * 100 <<"% ";
    }
    std::cout << std::endl;
*/
    for (int keep_ratio = start; keep_ratio <= end; keep_ratio += step) {
        
        int n_keep_points = (int)(keep_ratio / 100.0 * number_of_points);

        for (int i = 0; i < cell_num_vec.size(); i++) {
            if (n_keep_points >= cell_num_vec[i]) {

                std::vector<MYPoint> points_copy = points;
                points_copy.erase(CGAL::grid_simplify_point_set(points_copy.begin(), points_copy.end(), CGAL::Nth_of_tuple_property_map<0, MYPoint>(), cell_size_vec[i]),
                    points_copy.end());
                std::string dstFileSave = srcFilename + "." + std::to_string(keep_ratio);
                // std::cout << dstFileSave << std::endl;

                // std::cout << "ratio: " << 1.0 * points_copy.size() / number_of_points * 100 << std::endl;

                std::ofstream out(dstFileSave, std::ios_base::binary);
                for (int i = 0; i < points_copy.size(); i++) {
                        
                    float x = points_copy[i].get<0>().x();
                    float y = points_copy[i].get<0>().y();
                    float z = points_copy[i].get<0>().z();
                    float r = points_copy[i].get<1>();
                    float g = points_copy[i].get<2>();
                    float b = points_copy[i].get<3>();
                    float label = points_copy[i].get<4>();
                    out.write((char *)&x, sizeof(float));
                    out.write((char *)&y, sizeof(float));
                    out.write((char *)&z, sizeof(float));
                    out.write((char *)&r, sizeof(float));
                    out.write((char *)&g, sizeof(float));
                    out.write((char *)&b, sizeof(float));
                    out.write((char *)&label, sizeof(float));
                }
                out.close();
                break;
            }
        }
    }
}