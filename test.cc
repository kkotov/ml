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
    // require(MASS)
    // xyz <- mvrnorm( 1000000, c(0,0,0), matrix(c(1,0.7,0.5,0.7,1,0.7,0.5,0.7,1),ncol=3) )
    // plot(xyz[sample(nrow(xyz),10000),c(1,3)], xlab="x", ylab="y", pch=1)
    // write.csv(file="two.csv",x=xyz[sample(nrow(xyz),10000),])
    DataFrame df;
    ifstream input("two.csv");
    if( !input ) return df;
    setSeparators(input);
    readHeader(input,4);
    typedef tuple<string,float,float,string> Format;
    Format tmp;
    for(unsigned int row=0; read_tuple(input,tmp); row++){
        string level = get<3>(tmp);
        // strip quotes
        replace(level.begin(),level.end(),'\"',' ');
        tuple<float,float,int> r123 = make_tuple(
                get<1>(tmp), get<2>(tmp), stoi(level)
        );
        df.rbind( DataRow(r123) );
    }
    df.countAllLevels();
//    cout << "Found " << df.getLevels(2).size() << " levels:" << endl;
//    copy(df.getLevels(2).cbegin(),df.getLevels(2).cend(),ostream_iterator<int>(cout," "));
//    cout << endl;
    return df;
}



DataFrame readUltimate(void){
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

//    DataFrame df( read2() );
//    vector<unsigned int> predictorsIdx = {0,1,2,3,4,5};
//    rf.train(df,predictorsIdx,6);

    DataFrame df( read2() );
    DataFrame dfTrain, dfTest;
    for(size_t row=0; row<df.nrow(); row++)
        if( row%2 ) dfTrain.rbind(df[row]);
        else        dfTest. rbind(df[row]);
    dfTrain.countAllLevels();
    dfTest .countAllLevels();

    RandomForest rf;
    vector<unsigned int> predictorsIdx = {0,1};
    unsigned int responseIdx = 2;
    rf.train(dfTrain,predictorsIdx,responseIdx);

/*
    TreeTrainer tt;
    std::vector<unsigned int> shuffled = tt.sample(df.nrow(),df.nrow());
    Tree *tr = tt.findBestSplits(df, responseIdx, predictorsIdx, shuffled, false);
    Tree *tree = new Tree();
    tree->nodes.reserve(tr->tree_size);
    tr->vectorize(tree->nodes);

    std::shared_ptr<Tree> tree = tt.trainCART(df, predictorsIdx, responseIdx, 0);
    tree->save(cout);
    Tree rf = *tree;
*/

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
        cout << "bias = "<< bias << " sd = " << sd << " # events = " << cnt << endl;
    } else {
        map<long,map<long,unsigned int>> confusionMatrix;
        for(unsigned int row = 0; row>=0 && row < dfTest.nrow(); row++){
            long prediction = rf.classify( df[row] );
//            long prediction = rf.predict( df[row] ).asIntegral;
            long truth      = df[row][responseIdx].asIntegral;
            confusionMatrix[prediction][truth]++;
        }
        for(pair<const long,map<long,unsigned int>>& row : confusionMatrix){
            cout << row.first << ": " ;
            for(pair<const long,unsigned int> value : row.second)
                    cout << value.second <<", ";
            cout << endl;
        }
    }

    return 0;
}
