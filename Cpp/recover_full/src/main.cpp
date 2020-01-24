/* 
    creating labels using partial original labels
*/

#include <sys/types.h>
#include <dirent.h>

#include <iostream>
#include <fstream>

#define PCL_NO_PRECOMPILE
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>


struct MyPointType{
    PCL_ADD_POINT4D;
    int r;
    int g;
    int b;
    int label;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (MyPointType,           // here we assume a XYZ + "test" (as fields)
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (int, r, r)
                                   (int, g, g)
                                   (int, b, b)
                                   (int, label, label)
)


// list files in a directory
void list_files_in_dir(std::string folder, std::vector<std::string> &filenames, std::string suffix = "") {
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(folder.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.length() <  suffix.length() || filename.substr(filename.length() - suffix.length()).compare(suffix) != 0) {
                continue;
            }
            filenames.push_back(filename);
        }
    } else {
        std::cerr << "Cannot open " << folder << std::endl;
        exit(-1);
    }
}

void myReadPly(const std::string &file_name, pcl::PointCloud<MyPointType> &cloud) {
    int nSkipLine = 11; 

    std::ifstream infile(file_name.c_str());
    std::string line;

    for (int i = 0; i < nSkipLine; i++) {
        std::getline(infile, line);
    }

    float x, y, z; int r, g, b, label;  
    while (std::getline(infile, line)){
        std::istringstream iss(line);
        iss >> x >> y >> z >> r >> g >> b >> label;
        MyPointType point;
        point.x = x; point.y = y; point.z = z; point.r = r; point.g = g; point.b = b;
        point.label = label;
        cloud.push_back(point);
    }
}

void recover_full_from_partial(std::string ply_folder) {

    std::vector<std::string> ply_files;
    list_files_in_dir(ply_folder, ply_files, ".ply");

    for (auto item : ply_files) {
        std::string ply_file = ply_folder + "/" + item; 
        
        pcl::PointCloud<MyPointType> cloud_full, cloud_partial;
        myReadPly(ply_file, cloud_full);
        pcl::copyPointCloud(cloud_full, cloud_partial);

        for (int i = 1; i <= 100; i++) {

        }

        std::cout << cloud_partial.size() << std::endl;
    }
}

int main(int argc, char* argv[]) {
/*
Input:
    argv[0]: this program
    argv[1]: folder of ply files
*/

    if (argc != 2) {
        std::cerr << "[recover_full] requires argc: 2, input argc: " << argc << std::endl;
        std::cerr << "Format: program ply_folder" << std::endl;
        exit(-1);
    }

    std::string ply_folder(argv[1]);

    recover_full_from_partial(ply_folder);

    return 0;
}