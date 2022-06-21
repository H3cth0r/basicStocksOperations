#ifndef SOPS
#define SOPS

#include <fstream>

namespace sops{


    /*
        Struct for using dataframe data. The data on the csv's
        will be loaded to this struct.
    */
    template<int LEN>
    struct dataFrame{
        int    df_length = LEN;
        double Open[LEN];
        double High[LEN];
        double Low[LEN];
        double Close[LEN];
        double Adj_close[LEN];
        double Volume[LEN];
    };

    template<int LEN>
    int* split_line(std::string line){
        
    } 

    template<int LEN>
    dataFrame<LEN> read_csv(std::string csv_path){

        // Opening file
        std::fstream file(csv_path, std::ios::in);

        // Check if founded the file
        if(file.is_open()){

        } else{
            std::cout << "Unable to open/find the file."
        }
    }


}



#endif