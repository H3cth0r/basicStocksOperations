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

    return 0;
}