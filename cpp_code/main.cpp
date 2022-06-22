#include <iostream>
#include "SOps.h"


#define LENGTH 5

int main(){

    std::cout << "lol\n";

    sops::dataFrame<LENGTH> df;
    double up[LENGTH] = {32, 43, 54, 34, 53};
    for(int i = 0; i < LENGTH; i++){
        df.Open[i] = up[i];
    }
    for(int i = 0; i < LENGTH; i++){
        std::cout << df.Open[i] << ", ";
    }
    std::cout << "\n";

    std::string line = "2021-06-21,184.3524932861328,185.36500549316406,178.2274932861328,184.2725067138672,184.14715576171875,67238400";
    std::string something[7];
    std::string * res = sops::split_line<7>(line);
    
    for(int i = 0; i < 7; i++){
        std::cout << res[i] << ", ";
    }
    std::cout << "\n";

    return 0;
}