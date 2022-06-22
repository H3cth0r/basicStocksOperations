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
        std::string Date[LEN];
        double Open[LEN];
        double High[LEN];
        double Low[LEN];
        double Close[LEN];
        double Adj_close[LEN];
        double Volume[LEN];
    };

    template<int COLS>
    std::string* split_line(std::string line){
        std::string res[COLS];
        int pos_start = 0;
        int pos_end;
        int delim_len = 1;      // ",".length() == 1
        std::string token;
        int i = 0;
        
        while((pos_end=line.find(',', pos_start)) != std::string::npos){
            token = line.substr(pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;
            res[i] = token;
            std::cout << token << ", ";
            i++;
        }
        std::cout << "\n";
        
        return res;
    }
    // template<int LEN>
    // dataFrame<LEN> read_csv(std::string csv_path){
    //     std::string line;
    //     std::string line_to_array[LEN];

    //     // Opening file
    //     std::fstream file(csv_path, std::ios::in);

    //     // Check if founded the file
    //     if(file.is_open()){

    //     } else{
    //         std::cout << "Unable to open/find the file."
    //     }
    // }


}



#endif