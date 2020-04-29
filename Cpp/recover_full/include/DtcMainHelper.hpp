#pragma once

#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>

class DtcMainHelper {
private:
    static uint64_t programStartTimestamp;
    static std::ofstream dataOutstream;
private:
    
public: 
    static uint64_t getTimestamp();
    static std::ostream & formatCout();
    static std::ofstream & dataToFile(); 
    DtcMainHelper();
    ~DtcMainHelper();
};
