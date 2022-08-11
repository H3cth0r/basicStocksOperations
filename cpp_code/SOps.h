#ifndef SOPS
#define SOPS

#include <pthread.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <array>
#include <sstream>
#include <cmath>

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
      double VarianceReturns = 0;
      double StandardDeviation = 0;

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
       * Method for printing general information about the dataframe.
       */
      void printGeneralData(){
        std::cout << "Average Returns:\t" << AvgReturns <<
          "\n\t\t\t\t\t" << AvgReturns * 100 << "%" << 
          "\nVariance Returns:\t" << VarianceReturns <<
          "\n\t\t\t\t\t" << VarianceReturns * 100 << "%" <<
          "\nStandard Deviation:\t"<< StandardDeviation <<
          "\n";
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
     * Method for calculating the variance returns
     * @param  df  reference to dataframe working on.
     */
    template<int LEN>
    void varianceReturns(dataFrame<LEN> & df){
      for(int i = 0; i < LEN; i++){
        df.VarianceReturns += std::pow(df.Returns[i]-df.AvgReturns, 2);
      }
      df.VarianceReturns = df.VarianceReturns / LEN;
    }

    /*
     * @brief
     * Mathod for calculating the standard deviation.
     * @param  df  reference to dataframe working on.
     */
    template<int LEN>
    void stdDeviation(dataFrame<LEN> & df){
      df.StandardDeviation = std::pow(df.VarianceReturns, 0.5);
    }

    /*
     * @brief
     * Method for calculating the covariance between two sets of
     * stocks.
     * @param  dfX    reference to dataFrame X.
     *         dfY    reference to dataFrame Y.
     *         result reference to result variable.
     */
    template<int LEN>
    void covariance(dataFrame<LEN> & dfX, dataFrame<LEN> & dfY, double & result){
      for(int i = 0; i < LEN; i++){
        result += (dfX.Returns[i] * dfY.Returns[i]);
      }
      result = (result/LEN) - (dfX.AvgReturns * dfY.AvgReturns);
    }

    /*
     * @brief
     * Method for calculating the correlation between two sets of
     * stocks.
     * @param  dfX    reference to dataFrame X.
     *         dfY    reference to dataFrame Y.
     *         covar  reference to variable that contains covariance.
     *         result reference to variable for corr resutl.
     */
    template<int LEN>
    void correlation(dataFrame<LEN> & dfX, dataFrame<LEN> & dfY, double & covar,double & result){
      result = (covar) / (std::sqrt(dfX.VarianceReturns) * std::sqrt(dfY.VarianceReturns)); 
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
        //std::cout << "** Able to open file path **\n";
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

    /*
     * @brief
     * Method executes every necessary method to get the basic
     * and relevant information about a dataset.
     * @param    df      structure object of the dataframe.
     *           dfPath  name and path to the csv.
     */
    template<int LEN>
    void executeAll(dataFrame<LEN> & df, std::string & dfPath){
      df.Ticker = dfPath;
      readCsv(dfPath, df);
      closingReturns(df);
      averageReturns(df);
      varianceReturns(df);
      stdDeviation(df);
    }

    /*
     * @brief
     * Method that will get multithreaded, which takes a range of 
     * the array of dataFrames and paths.
     *
     * @param  min    specifies the min number of the range.
     *
     *         max    specifies the max number of the range.
     *
     *         dfs    reference to dataFrames structures array.
     *
     *         dfsPaths reference to array of the paths.
     */
    template<int NumDfs, int LEN>
    void multiThreadSol(int min, int max, std::array<dataFrame<LEN>, NumDfs> & dfs, std::array<std::string, NumDfs> & dfsPaths){
      //std::cout << "min: " << min << "\tmax: " << max << "\n";
      for(int i = min; i < max; i++){
        //std::cout << "path: " << dfsPaths[i] << "\n";
        executeAll(dfs[i], dfsPaths[i]);
      }
      
    }
  
    /*
     * @brief
     * Method for reading many dataframes and creating many
     * dataframe structures. Applies multiple threads for 
     * a faster implementation.
     *
     * @param    dfsPaths      array of file names and paths
     *                         for reading and opening the cvs.
     *            
     *
     * @param    dfs            reference to array that contains the
     *                          dataFrame structures.
     *
     * @param    numThreads     Number of threads to execute the 
     *                          program.
     * 
     */
    template<int NumDfs, int LEN>
    void manyDfs(std::array<std::string, NumDfs> & dfsPaths, std::array<dataFrame<LEN>, NumDfs> & dfs, int numThreads){
      std::array<std::thread, NumDfs> threads;
      int steps = NumDfs/numThreads;
      int aux_min = 0;
      for(int i = 0; i < numThreads; i++){
        threads[i] = std::thread(multiThreadSol<NumDfs, LEN>, aux_min, aux_min + steps, std::ref(dfs), std::ref(dfsPaths));
        aux_min += steps;
      }
      for(int i = 0; i < numThreads; i++){
        threads[i].join();
      }
    }
}


#endif