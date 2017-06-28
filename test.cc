#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>

#include "DataFrame.h"
#include "RandomForest.h"

// g++ -Wl,--no-as-needed -g -Wall -std=c++11 -o rf test.cc -lpthread

using namespace std;

DataFrame oneHOTencode(const vector<int> &col, unordered_set<int> levels = {}, bool nMinusOne = true){
    DataFrame df;
    // if levels are not provided, deduce them; provided two - check
    if( levels.size() <= 2 ){
        levels.clear();
        copy(col.cbegin(), col.cend(), inserter(levels,levels.begin()));
    }
    // do nothing for binary levels
    if( levels.size() == 2 ){
        df.cbind(col);
        return df;
    }
    // encode
    for(auto l: levels){
        // N-1 levels? - make 0th level - all zeros
        if( nMinusOne && l == *levels.begin() ) continue;
        // turn each level into a binary match/mismatch column
        vector<char> binary( col.size() );
        transform(col.cbegin(), col.cend(), binary.begin(), [&l] (int i){ return i==l; } );
        df.cbind(binary);
    }
    return df;
}

template<int IDX, int NMAX, typename... Args>
struct READ_TUPLE {
    static bool read(istream &in, tuple<Args...> &t){
        if( in.eof() ) return false;
        in >> get<IDX>(t);
        return READ_TUPLE<IDX+1,NMAX,Args...>::read(in,t);
    }
};
template<int NMAX, typename... Args>
struct READ_TUPLE<NMAX,NMAX,Args...>{
    static bool read(istream &in, tuple<Args...> &t){ return true; }
};
template <typename... Args>

bool read_tuple(istream &in, tuple<Args...> &t) noexcept {
    return READ_TUPLE<0,sizeof...(Args),Args...>::read(in,t);
}

void setSeparators(istream& input){
    struct field_reader: std::ctype<char> {
        field_reader(): std::ctype<char>(get_table()) {}

        static std::ctype_base::mask const* get_table() {
            static std::vector<std::ctype_base::mask> 
                rc(table_size, std::ctype_base::mask());

            rc['\n'] = std::ctype_base::space;
            rc[':']  = std::ctype_base::space;
            rc[',']  = std::ctype_base::space;
            return &rc[0];
        }
    };
    input.imbue(std::locale(std::locale(), new field_reader()));
}

void readHeader(istream& input, unsigned int ncol){
    unordered_map<string,unsigned int> dict;
    class my_dict_output_iterator : public iterator<output_iterator_tag,typename unordered_map<string,unsigned int>::value_type> {
        private:
            unsigned int counter;
        protected:
            unordered_map<string,unsigned int>& container;
        public:
            explicit my_dict_output_iterator(unordered_map<string,unsigned int> &c) : counter(0), container(c){ }
            my_dict_output_iterator operator= (const string& str){
                container.insert( make_pair(str,counter++) );
                return *this;
            }
            my_dict_output_iterator& operator*  (void){ return *this; }
            my_dict_output_iterator& operator++ (void){ return *this; }
            my_dict_output_iterator& operator++ (int) { return *this; }
    };
    copy_n(istream_iterator<string>(input), ncol, my_dict_output_iterator(dict));
}

DataFrame read1(void){
    // require(MASS)
    // xy <- mvrnorm( 1000000, c(1,2), matrix(c(3,2,2,4),ncol=2) )
    // plot(xy[sample(nrow(xy),10000),], xlab="x", ylab="y", pch=1)
    // write.csv(file="one.csv",x=xy[sample(nrow(xy),10000),])
    DataFrame df;
    ifstream input("one.csv");
    if( !input ) return df;
    setSeparators(input);
    readHeader(input,3);
    typedef tuple<string,float,float> Format;
    Format tmp;
    for(unsigned int row=0; read_tuple(input,tmp); row++){
        tuple<float,float> r12 = make_tuple(
                get<1>(tmp), get<2>(tmp)
        );
        df.rbind( DataRow(r12) );
    }
    df.countAllLevels();
    return df;
}

DataFrame read2(void){
    DataFrame df;
    ifstream input("../trigger/pt/SingleMu_Pt1To1000_FlatRandomOneOverPt.csv");
    if( !input ) return df;
    setSeparators(input);
    readHeader(input,53);
    typedef tuple<int,float,float,float,int,float,float,float,float,float,float,int,int,int,int,int,int,int,int,int,int,int,int,int,int,
                  int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int> Format;
#define dPhi12_0 13
#define dPhi12_1 14
#define dPhi23_0 19
#define dPhi23_1 20
#define dPhi34_0 23
#define dPhi34_1 24
#define dPhi13_0 15
#define dPhi13_1 16
#define dPhi14_0 17
#define dPhi14_1 18
#define dPhi24_0 21
#define dPhi24_1 22
#define muPtGen  1
    Format tmp;
    for(unsigned int row=0; read_tuple(input,tmp); row++){
        if( get<11>(tmp) == 15 ){
            tuple<float,float,float,float,float,float,float> dPhis = make_tuple(
                abs(get<dPhi12_0>(tmp)), abs(get<dPhi23_0>(tmp)), abs(get<dPhi34_0>(tmp)),
                abs(get<dPhi13_0>(tmp)), abs(get<dPhi14_0>(tmp)), abs(get<dPhi24_0>(tmp)),
                1./get<muPtGen>(tmp)
            );
            df.rbind( DataRow(dPhis) );
        }
        if( get<12>(tmp) == 15 ){
            tuple<float,float,float,float,float,float,float> dPhis = make_tuple(
                abs(get<dPhi12_1>(tmp)), abs(get<dPhi23_1>(tmp)), abs(get<dPhi34_1>(tmp)),
                abs(get<dPhi13_1>(tmp)), abs(get<dPhi14_1>(tmp)), abs(get<dPhi24_1>(tmp)),
                1./get<muPtGen>(tmp)
            );
            df.rbind( DataRow(dPhis) );
        }
    }
    df.countAllLevels();
    return df;
}


int main(void){
    RandomForest rf;

//    DataFrame df( read2() );
//    vector<unsigned int> predictorsIdx = {0,1,2,3,4,5};
//    rf.train(df,predictorsIdx,6);

    DataFrame df( read1() );
    vector<unsigned int> predictorsIdx = {0};
    DataFrame dfTrain, dfTest;
    for(size_t row=0; row<df.nrow(); row++)
        if( row%2 ) dfTrain.rbind(df[row]);
        else        dfTest. rbind(df[row]);
    df.countAllLevels();
    rf.train(dfTrain,predictorsIdx,1);

//    rf.ensemble[0].save(cout);

    double bias = 0, var = 0;
    long cnt = 0;
    for(unsigned int row = 0; row>=0 && row < dfTest.nrow(); row++,cnt++){
        double prediction = rf.regress( df[row] );
        double truth      = df[row][1].asFloating; // 6
// cout << df[row] <<endl;
//        cout << "prediction = "<<prediction <<" truth= "<<truth<<endl;
//        double prediction = 1./rf.regress( df[row] );
//        double truth      = 1./df[row][6].asFloating;
        bias +=  prediction - truth;
        var  += (prediction - truth) * (prediction - truth);
    }
    double sd = sqrt((var - bias*bias/cnt)/(cnt - 1));
    bias /= cnt;
    cout << "bias = "<< bias << " sd = " << sd << endl;

    return 0;
}
