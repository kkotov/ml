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
    csvUtils::setCommaDelim(input);

    // specify file format
    std::tuple<std::string,float,float,std::string> format;

    // read file to the end
    while( csvUtils::read_tuple(input,format) )
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

    // train random forest to predict V3 using V1 and V2
    const int V1 = 0, V2 = 1, V3 = 2;
    RandomForest rf1;
    std::vector<unsigned int> predictorsIdx = {V1,V2};
    unsigned int responseIdx = V3;
    rf1.train(dfTrain,predictorsIdx,responseIdx);

    // see how well classification is doing
    std::map<long,std::map<long,unsigned int>> confusionMatrix;
    for(unsigned int row = 0; row>=0 && row < dfTest.nrow(); row++){
        long prediction = rf1.classify( dfTest[row] );
        long truth      = dfTest[row][responseIdx].asIntegral;
        confusionMatrix[prediction][truth]++;
    }

    // print column levels
    std::cout << "Classification performance: " << std::endl << "\t";
    for(std::pair<const long,unsigned int> i : confusionMatrix.begin()->second)
        std::cout << i.first << "\t";
    std::cout << std::endl;
    // print matrix itself
    for(std::pair<const long,std::map<long,unsigned int>>& row : confusionMatrix){
        std::cout << row.first << ":\t" ;
        for(std::pair<const long,unsigned int> value : row.second)
            std::cout << value.second <<"\t";
        std::cout << std::endl;
    }


    // A simple unit test for the IO
    std::ofstream file1("rf1.model");
    rf1.save(file1);
    file1.close();

    std::ifstream file2("rf1.model");
    RandomForest ioTest;
    ioTest.load(file2);
    file2.close();
    
    std::ofstream file3("_rf1.model");
    ioTest.save(file3);
    file3.close();
    // diffing rf1.model and _rf1.model shows no difference


    // train another random forest to predict V1 using V2 and V3
    RandomForest rf2;
    predictorsIdx = {V2,V3};
    responseIdx = V1;
    rf2.train(dfTrain,predictorsIdx,responseIdx);

    std::cout << std::endl << "Regression performance: " << std::endl;
    double bias = 0, var = 0;
    long cnt = 0;
    for(unsigned int row = 0; row>=0 && row < dfTest.nrow(); row++,cnt++){
        double prediction = rf2.regress( dfTest[row] );
        double truth      = dfTest[row][responseIdx].asFloating;
        bias +=  prediction - truth;
        var  += (prediction - truth) * (prediction - truth);
    }
    double sd = sqrt((var - bias*bias/cnt)/(cnt - 1));
    bias /= cnt;
    std::cout << "bias = "<< bias << " sd = " << sd << std::endl;

    return 0;
}
