#ifndef SOPS
#define SOPS

#include <iostream>
#include <fstream>
#include <array>
#include <sstream>

namespace ezfunc{

  void print(std::string value){
    std::cout << value << "\n";
  }
}

namespace sops{
    
    /*
     * @brief
     * Struct for creating a dataframe dedicated to tickers
     * data. The datatype implemented for storing the colum-
     * ns are modern std::arrays that have extra tools with
     * the same efficiency.
     *
     * @param  LEN          number of rows un dataframe.
     */
    template<int LEN>
    struct dataFrame{
      std::string Ticker;
      std::array<std::string, LEN> Date;
      std::array<double, LEN> Open;
      std::array<double, LEN> High;
      std::array<double, LEN> Low;
      std::array<double, LEN> Close;
      std::array<double, LEN> Adj_close;
      std::array<double, LEN> Volumen;
      std::array<double, LEN> Returns;

      double AvgReturns = 0;

      /*
       * @brief
       * Method for printing the values in each row, depending
       * of number of values desired to display.
       * @param  min  min location of rows to print.
       * @param  max  max location of rows to print.
       */
      void print(int min = 0, int max = 6){
        for(int i = min; i < max; i++){
          std::cout << "Date:\t\t\t" << Date[i] <<"\nOpen:\t\t\t" << Open[i] 
            <<  "\nHigh:\t\t\t" << High[i] << "\nLow:\t\t\t" << Low[i] << 
            "\nAdj_close:\t\t" << Adj_close[i] << "\nVolumen:\t\t"<<
            Volumen[i] << "\nReturns:\t\t" << Returns[i] <<
            "\n=================\n";
        }
      }

      /*
       * @brief
       * Method for printing the average returns on the df.
       */
      void printAvgReturns(){
        std::cout << "Average Returns\nDecimal:\t"
        << AvgReturns << "\nPercentage:\t" << AvgReturns*100 << "%\n";  
      }
    };

    /*
     * @brief
     * Method for calculating and setting values to the returns
     * column in the dataframe.
     * @param  df  reference to the dataframe that is being work on.
     */
    template<int LEN>
    void closingReturns(dataFrame<LEN> & df){
      for(int i = 0; i - LEN; i++){
        if(i < LEN-1){
          df.Returns[i] = df.Adj_close[i]/df.Adj_close[i+1]-1;
        } else{
          df.Returns[i] = 0;
        }
      }
    }

    /*
     * @brief
     * Method for calculating the average returns in the df.
     * @param  df  reference to dataframe working on.
     */
    template<int LEN>
    void averageReturns(dataFrame<LEN> & df){
      df.AvgReturns = 0;
      for(int i = 0; i < LEN; i++){
        df.AvgReturns += df.Returns[i];
      }
      df.AvgReturns = df.AvgReturns/LEN;
    }

    /*
     * @brief
     * Method for reading the word into the line. Sets the values
     * in corresponding into column.
     * @param  line    reference to the line being read.
     * @param  df      dataframe being applied.
     * @param  rowNum  iterator for tracking row.
     * 
     */
    template<int LEN>
    void readSingleLine(std::string & line, dataFrame<LEN> & df, int & rowNum){
      std::stringstream str(line);
      std::string word;
      int columnIterator = 0;
      while(getline(str, word, ',')){
        switch(columnIterator){
          case 0:
            df.Date[rowNum] = word;
            break;
          case 1:
            df.Open[rowNum] = std::stod(word);
            break;
          case 2:
            df.High[rowNum] = std::stod(word);
            break;
          case 3:
            df.Low[rowNum] = std::stod(word);
            break;
          case 4:
            df.Close[rowNum] = std::stod(word);
            break;
          case 5:
            df.Adj_close[rowNum] = std::stod(word);
            break;
          case 6:
            df.Volumen[rowNum] = std::stod(word);
            break;
        }
        columnIterator ++;
      }
    }

    /*
     * @brief
     * Method for reading line by line. Uses readSingleLine
     * for reading the values into the line.
     * @param    file      reference to the file being read.
     * @param    df        reference of dataframe being used.
     */
    template<int LEN>
    void readLines(std::fstream & file, dataFrame<LEN> & df){
      std::string line;
      bool firstLine = true;
      int rowNum = 0;
      while(getline(file, line)){
        if(!firstLine){
          readSingleLine(line, df, rowNum);
          rowNum++;
        }
        else firstLine = false;
      }
    }

    /*
     * @brief
     * Method for giving the instruction to read the csv file.
     * @param  fileNameNPath      the name and location system
     *                            of the file to be read.
     * @param  df                 reference to the struct da-
     *                            frame were data is input.
     *
     */
    template<int LEN>
    void readCsv(std::string fileNameNPath, dataFrame<LEN> & df){
      std::fstream file(fileNameNPath);
      if(file.is_open()){
        std::cout << "** Able to open file path **\n";
        readLines(file, df);
      }else{
        std::cout << "File not found or error on path **\n";
      }
    }

    /*
     * @brief
     * Method for iterating lines.
     * @param  file    reference to opened file.
     */
    int countLines(std::fstream & file){
      std::string line;
      int counter = 0;
      while(getline(file, line)){
        counter ++;
      }
      return counter - 1;
    }

    /*
     * @brief
     * Method for counting the number of rows in csv or txt file.
     * @param  fileNamePath    Name and path of the file.
     */
    int numberOfRowsInFile(std::string fileNameNPath){
      // Opening file
      std::fstream file(fileNameNPath);

      if(file.is_open()){
        std::cout << "** Able to open file path **\n";
        return countLines(file);
        
      }else{
        std::cout << "File not found or error on path **\n";
      }
      return 0;
    }

}


#endif