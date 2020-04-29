#include "DtcMainHelper.hpp"
#include <chrono>

uint64_t initializeTimestamp(){
    std::chrono::high_resolution_clock t_now;
    return std::chrono::duration_cast<std::chrono::microseconds> 
                                    (t_now.now().time_since_epoch()).count();
}

std::ofstream initializeOutstream(std::string filename) {
    std::ofstream outFile(filename.c_str(), std::ios::out | std::ofstream::app);
    return outFile;
}

uint64_t DtcMainHelper::getTimestamp() {
    std::chrono::high_resolution_clock t_now;
    uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds> 
                                    (t_now.now().time_since_epoch()).count();
    return timestamp - DtcMainHelper::programStartTimestamp;
}

std::ostream & DtcMainHelper::formatCout() {
    std::cout << std::setw(9) << DtcMainHelper::getTimestamp() << " (us)  ";
    return std::cout;
}

std::ofstream & DtcMainHelper::dataToFile() {
    return DtcMainHelper::dataOutstream;
}

DtcMainHelper::DtcMainHelper() {

}

DtcMainHelper::~DtcMainHelper() {
    DtcMainHelper::dataOutstream.close();
}

// initialize static variables
uint64_t DtcMainHelper::programStartTimestamp = initializeTimestamp();
std::ofstream DtcMainHelper::dataOutstream = initializeOutstream(std::string("../tmp/time.txt"));