#include <iostream>
#include <fstream>
#include <tuple>
#include <string>

#include "DataFrame.h"
#include "RandomForest.h"
#include "csvUtils.h"

// g++ -Wl,--no-as-needed -g -Wall -std=c++11 -o rf example.cc -lpthread

int main(void){
    // data for training and testing
    DataFrame df;

    // input file
    std::ifstream input("one.csv");
    if( !input ) return 0;

    // read first line that is the header
    std::string line;
    std::getline(input,line);

    // treat comma as a delimiter
    setCommaDelim(input);

    // specify file format
    std::tuple<std::string,float,float,std::string> format;

    // read file to the end
    while( read_tuple(input,format) )
    {
        std::string level = std::get<3>(format);
        // strip quotes
        std::replace(level.begin(),level.end(),'\"',' ');
        std::tuple<float,float,int> row = std::make_tuple(
                std::get<1>(format), std::get<2>(format), std::stoi(level)
        );
        df.rbind( DataRow(row) );
    }
    // countAllLevels has to be called in the end of reading input with categorical variables
    df.countAllLevels();

    // split the data frame into training and test data partitions:
    DataFrame dfTrain, dfTest;
    for(size_t row=0; row<df.nrow(); row++)
        if( row%2 ) dfTrain.rbind(df[row]);
        else        dfTest. rbind(df[row]);
    dfTrain.countAllLevels();
    dfTest .countAllLevels();

    // train random forest
    RandomForest rf;
    std::vector<unsigned int> predictorsIdx = {0,1};
    unsigned int responseIdx = 2;
    rf.train(dfTrain,predictorsIdx,responseIdx);

    // benchmarking regression of classification?
    if( df.getLevels(responseIdx).size() == 0 ){
        double bias = 0, var = 0;
        long cnt = 0;
        for(unsigned int row = 0; row>=0 && row < dfTest.nrow(); row++,cnt++){
            double prediction = rf.regress( df[row] );
            double truth      = df[row][responseIdx].asFloating;
            bias +=  prediction - truth;
            var  += (prediction - truth) * (prediction - truth);
        }
        double sd = sqrt((var - bias*bias/cnt)/(cnt - 1));
        bias /= cnt;
        std::cout << "bias = "<< bias << " sd = " << sd << " # events = " << cnt << std::endl;
    } else {
        std::map<long,std::map<long,unsigned int>> confusionMatrix;
        for(unsigned int row = 0; row>=0 && row < dfTest.nrow(); row++){
            long prediction = rf.classify( df[row] );
            long truth      = df[row][responseIdx].asIntegral;
            confusionMatrix[prediction][truth]++;
        }
        for(std::pair<const long,std::map<long,unsigned int>>& row : confusionMatrix){
            std::cout << row.first << ": " ;
            for(std::pair<const long,unsigned int> value : row.second)
                    std::cout << value.second <<", ";
            std::cout << std::endl;
        }
    }

    return 0;
}
