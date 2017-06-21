#include <iostream>
#include <fstream>
#include <tuple>
#include <list>
#include <vector>
#include <iterator>
#include <algorithm>
#include <string>
#include <random>
#include <unordered_set>
#include <unordered_map>
using namespace std;

// g++ -Wl,--no-as-needed -g -Wall -std=c++11 -o rf rf.cc -lpthread

// No better solution for the type-obsessed languages but to create a type
//  that can play for the both sides
struct Variable {
    enum Type { Unknown=0, Categorical=1, Continuous=2 }; // Categorical is always considered unordered below
    Type type;
    union {
        long long asIntegral;
        double    asFloating;
    };
    friend ostream& operator<< (ostream&, const Variable&);
    ostream& operator<< (ostream& out) const {
        switch(type){
            case Categorical: out << "(int)"   << asIntegral; break;
            case Continuous:  out << "(float)" << asFloating; break;
            default : out << "Unkown"; break;
        }
        return out;
    }
    Variable(void){ type = Unknown; asIntegral = 0; }
    explicit Variable(long long integral){ type = Categorical; asIntegral = integral; }
    explicit Variable(double    floating){ type = Continuous;  asFloating = floating; }
};

ostream& operator<< (ostream& out, const Variable& var) { return var.operator<<(out); }

// The DataRow abstraction is meant to be an interface between templated and
//  non-templated worlds. Although the whole RandomForests framework could
//  have been made templated, this brings an unnecessary generalization and
//  blows the code out of proportion, while in fact we only deal with two types
//  of variables: categorical (integral) and non-categorical (floating point)
class DataRow {
private:
    // all the elements of the row are stored in this vector
    vector<Variable> data;

    // helper: store element with index IDX of tuple with MAX elements
    template <int IDX, int MAX, typename... Args>
    struct STORE_TUPLE {
        static void store(vector<Variable>& d, const std::tuple<Args...>& t) {
            auto element = std::get<IDX>(t);
            if( std::is_integral<decltype(element)>::value )
                d.push_back( Variable((long long) element ) );
            else
                d.push_back( Variable((double   ) element) );
            STORE_TUPLE<IDX+1,MAX,Args...>::store(d,t);
        }
    };

    // partial specialization to end the recursion
    template <int MAX, typename... Args>
    struct STORE_TUPLE<MAX,MAX,Args...> {
        static void store(vector<Variable>& d, const std::tuple<Args...>& t) {}
    };

    friend class DataFrame;

public:

    // interfacing DataRow with tuple
    template <typename... Args>
    DataRow& operator=(const tuple<Args...> &t) noexcept {
        data.resize(0);
        STORE_TUPLE<0,sizeof...(Args),Args...>::store(data,t);
        return *this;
    }

    template <typename... Args>
    DataRow(const tuple<Args...> &t) noexcept {
        STORE_TUPLE<0,sizeof...(Args),Args...>::store(data,t);
    }

    // subsetting
          Variable& operator[](unsigned int i)       { return data[i]; }
    const Variable& operator[](unsigned int i) const { return data[i]; }

    ostream& operator<< (ostream& out) const {
        copy(data.cbegin(),data.cend(),ostream_iterator<Variable>(out," "));
        return out;
    }

    DataRow(void){}
    // copy, and move c-tors will be generated by the compiler
};

ostream& operator<< (ostream& out, const DataRow& dr) { return dr.operator<<(out); }

// abstraction for grouping DataRows together
class DataFrame {
private:
    vector<unsigned int> schema; // 1 - continuous, >=2 - number of levels in categorical
    vector<DataRow> rows;

public:
    const vector<unsigned int>& getSchema(void) const { return schema; }
    unsigned int nrow(void) const { return rows.size(); }
    unsigned int ncol(void) const { return schema.size(); }

    template<typename T>
    bool cbind(const vector<T> &col, unsigned int nLevels=1) {
        // check if number of rows matchs number of elements in column
        if( col.size() != rows.size() && rows.size() != 0 )
            return false;
        // in case the DataFrame is empty initialize it with the column
        if( rows.size() == 0 ) rows.resize( col.size() );
        // the two options: categorical/integral and floating/continuous
        if( std::is_integral<T>::value ){
            for(unsigned i=0; i<col.size(); ++i)
                rows[i].data.emplace_back((long long)col[i]);
            // deduce number of levels automatically
            unordered_set<long long> unique;
            copy(col.cbegin(), col.cend(), inserter(unique,unique.begin()));
            // store number of found or provided levels
            if( nLevels > unique.size() )
                schema.push_back( nLevels );
            else
                schema.push_back( unique.size() );
        } else {
            for(unsigned i=0; i<col.size(); ++i)
                rows[i].data.emplace_back((double)col[i]);
            // mark the column as continuous
            schema.push_back(1);
        }
        return true;
    }

    bool rbind(DataRow && row) {
        // check if number of elements in the row agrees with the expectation
        if( row.data.size() != schema.size() && schema.size() > 0 )
            return false;
        // check if we start fresh
        if( schema.size() == 0 ){
            // initialize the empty DataFrame with the row
            rows.push_back(row);
            transform(row.data.cbegin(), row.data.cend(), back_inserter(schema),
                [](const Variable& var){ return (var.type == Variable::Categorical ? 2 : 1) ; }
            );
        } else {
            // make sure we preserve the schema but do nothing about number of levels assuming it includes the binded
            if( !equal(row.data.cbegin(), row.data.cend(), schema.cbegin(),
                     [](const Variable& var, int type){
                         return (var.type == Variable::Categorical && type >= 2) ||
                                (var.type == Variable::Continuous  && type == 1) ;
                     }
                 )
            ) return false;
        }
        rows.push_back( move(row) );
        return true;
    }

          DataRow& operator[](unsigned int i)       { return rows[i]; }
    const DataRow& operator[](unsigned int i) const { return rows[i]; }

    ostream& print(ostream& out, int nrows=-1) const {
        copy( rows.cbegin(),
              (nrows<0 ? rows.cend() : rows.cbegin()+nrows),
              ostream_iterator<DataRow>(out,"\n")
        );
        return out;
    }
};

ostream& operator<<(ostream& out, const DataFrame& df){ return df.print(out); }

class Tree {
private:
    struct Node {
        Variable value;
        int position;
        int left_child, right_child;
        Node(void) : value(),  position(0), left_child(0), right_child(0){}
        explicit Node(double v) : value((double)    v), position(0), left_child(0), right_child(0){}
        explicit Node(long   v) : value((long long) v), position(0), left_child(0), right_child(0){}
    };
    friend class RandomForest;
    vector<Node> nodes;

public:
    Variable traverse(const DataRow& row, const Node& root) const {
        // is it a leaf/terminal_node?
        if( root.left_child == 0 && root.right_child == 0 )
            return root.value;

        if( root.value.type == Variable::Continuous ){
            if( root.value.asFloating > row[root.position].asFloating )
                return traverse(row,nodes[root.left_child]);
            else
                return traverse(row,nodes[root.right_child]);
        }
        if( root.value.type == Variable::Categorical ){
            // only binary-level categories are managed
            if( root.value.asIntegral == row[root.position].asIntegral )
                return traverse(row,nodes[root.left_child]);
            else
                return traverse(row,nodes[root.right_child]);
        }
        // the root is neither Continuous nor Categorical -> error
        return Variable();
    }
    Variable predict(const DataRow& row) const {
        // is tree initialized? if not return default Variable as a sign of error
        if( nodes.size() == 0 ) return Variable(); 
        // is root node initialized?
        if( nodes[0].value.type == Variable::Unknown ) return Variable(); 
        // all looks good
        return traverse(row,nodes[0]);
    }

    bool load(istream &input){
        return true;
    }
    bool save(ostream &output){
        for(unsigned int n=0; n<nodes.size(); n++)
            output << n
                   << "," << nodes[n].value
                   << "," << nodes[n].position
                   << "," << nodes[n].left_child
                   << "," << nodes[n].right_child
                   << endl;
        return true;
    }
};

class RandomForest {
private:
public:
    typedef list<pair<unsigned int,unsigned int>> SplitVars; // variable index, level (if categorical)

    std::default_random_engine rState;

    SplitVars generateRandomSplitVars(const vector<unsigned int> &schema, const vector<unsigned int>& predictorsIdx, unsigned int mtry){
        SplitVars vars;
        default_random_engine dre(rState);
        uniform_int_distribution<unsigned int> uid(0, predictorsIdx.size()-1), uid_l;
        generate_n( back_inserter(vars),
                    mtry,
                    [&uid,&uid_l,&dre,&schema,&predictorsIdx](void){
                        unsigned int idx = predictorsIdx[ uid(dre) ];
                        unsigned int level = (schema[idx]>1 ? uid_l(dre)%schema[idx] : 0);
                        return pair<unsigned int,unsigned int>(idx,level);
                    }
        );
        return vars;
    }

    vector<unsigned int> sample(unsigned int nTotal, unsigned int nSampled, bool replace = false){
        // definitely, there is room for improvement below
        vector<unsigned int> retval(nTotal);
        if( !replace ){
            iota(retval.begin(),retval.end(),0);
            shuffle(retval.begin(),retval.end(),rState);
        } else {
            default_random_engine dre(rState);
            uniform_int_distribution<> uid(0, nTotal);
            generate_n( retval.begin(), (nSampled<nTotal?nSampled:nTotal), [&uid,&dre](void){ return uid(dre); } );
        }
        return vector<unsigned int>(retval.begin(), retval.begin() + (nSampled<nTotal?nSampled:nTotal));
    }

    // simplest forward-stepwise
    Tree findBestSplits(const DataFrame& df,
                        unsigned int responseIdx,
                        const SplitVars& vars,
                        const vector<unsigned int>& subset = {}
    ){
        // LDA for categorical response?
        // other metrics: purity/gini/entrophy/rms

        if( subset.size() == 0 ) return Tree();

        // end of recursion
        if( vars.empty() ){
            double median_cut = 0, sum = 0;
            if( df.getSchema()[responseIdx] == 1 ){
                unsigned int size = subset.size();
                vector<double> vals(size);
                for(unsigned int i=0; i<size; ++i){
                    vals[i] = df[subset[i]][responseIdx].asFloating;
                    sum += vals[i];
                }
                nth_element(vals.begin(), vals.begin() + size/2, vals.end());
                median_cut = vals[size/2];
            }
            Tree leaf;
            leaf.nodes.push_back( Tree::Node( sum/subset.size() ) ); //median_cut ) ); //
            return leaf;
        }

        // for continuous response simply calculate correlations and pick the strongest; ignore the outliers?
//        if( df[0][responseIdx].type == Variable::Continuous ){
            // from https://github.com/koskot77/clustcorr/blob/master/src/utilities.cc
            unsigned int size = subset.size();

            double varTarget = 0, meanTarget = 0;
            for(unsigned int i=0; i<size; ++i){
                unsigned int row = subset[i];
                varTarget  += df[row][responseIdx].asFloating * df[row][responseIdx].asFloating;
                meanTarget += df[row][responseIdx].asFloating;
            }
            double sdTarget = sqrt((varTarget - meanTarget*meanTarget/size)/(size - 1));
            meanTarget /= size;

            struct HashPair { size_t operator()(const pair<unsigned int,unsigned int>& p) const { return p.first*10 + p.second; } };
            unordered_map<pair<unsigned int,unsigned int>, double, HashPair> corr;
            for(pair<unsigned int,unsigned int> pick : vars){
                double var = 0, crossVar = 0, mean = 0;
                for(unsigned int i=0; i<size; ++i){
                    unsigned int row = subset[i];
                    if( df.getSchema()[ pick.first ] == 1 ){
                        mean     += df[row][pick.first].asFloating;
                        var      += df[row][pick.first].asFloating * df[row][pick.first].asFloating;
                        crossVar += df[row][pick.first].asFloating * df[row][responseIdx].asFloating;
                    } else {
                        mean     += (df[row][pick.first].asIntegral == pick.second);
                        var      += (df[row][pick.first].asIntegral == pick.second);
                        crossVar += (df[row][pick.first].asIntegral == pick.second) * df[row][responseIdx].asFloating;
                    }
                }
                double sd = sqrt((var - mean*mean/size)/(size - 1));
                mean /= size;
                corr[pick] = (sd>0 ? size/double(size-1) * (crossVar/size-meanTarget*mean) / sdTarget / sd : 0);
            }

            unordered_map<pair<unsigned int,unsigned int>, double, HashPair>::const_iterator bestCut = 
                max_element(corr.cbegin(),
                            corr.cend(),
                            [](pair<pair<unsigned int,unsigned int>, double> a, pair<pair<unsigned int,unsigned int>, double> b){
                                return a.second < b.second;
                            }
                );

//cout << " var= " << bestCut->first.first << " idx= " << bestCut->first.second << " corr= " << bestCut->second << " vars.size() " << vars.size() << endl;

            SplitVars remainingSplitVars;
            bool once = false;
            copy_if( vars.cbegin(),
                     vars.cend(),
                     back_inserter(remainingSplitVars),
                     [&bestCut,&once](pair<unsigned int,unsigned int> s){
                         if( !once ){
                             bool match = s.first != bestCut->first.first || s.second != bestCut->first.second;
                             if( !match ) once = true;
                             return match;
                         } else
                             return true;
                     }
            );

            vector<unsigned int> left_subset, right_subset;
            double median_cut = 0, sum = 0;
            if( df.getSchema()[bestCut->first.first] == 1 ){
                unsigned int size = subset.size();
                vector<double> vals(size);
                for(unsigned int i=0; i<size; ++i){
                    vals[i] = df[subset[i]][bestCut->first.first].asFloating;
                    sum += vals[i];
                }
                nth_element(vals.begin(), vals.begin() + size/2, vals.end());
                median_cut = vals[size/2];
                sum /= size;
            }
            for(unsigned int i : subset)
                switch(  df[i][bestCut->first.first].type ){
                    case Variable::Continuous :
                        if( df[i][bestCut->first.first].asFloating < median_cut )
                            left_subset.push_back(i);
                        else
                            right_subset.push_back(i);
                    break ;
                    case Variable::Categorical:
                        if( df[i][bestCut->first.first].asIntegral == bestCut->first.second )
                            left_subset.push_back(i);
                        else
                            right_subset.push_back(i);
                    break ;
                    default : return Tree(); break;
                }

            // good place to use the thread pool
            Tree left_subtree  = findBestSplits(df, responseIdx, remainingSplitVars, left_subset);
            Tree right_subtree = findBestSplits(df, responseIdx, remainingSplitVars, right_subset);

            Tree tree;
            tree.nodes.resize(1 + left_subtree.nodes.size() + right_subtree.nodes.size());
            // copy the local root node
            Tree::Node& local_root = tree.nodes[0];
            local_root.position = bestCut->first.first;
            if( df.getSchema()[bestCut->first.first] == 1 ){
                local_root.value.type = Variable::Continuous;
                local_root.value.asFloating = median_cut;
            } else {
                local_root.value.type = Variable::Categorical;
                local_root.value.asIntegral = bestCut->first.second;
            }
            local_root.left_child = (left_subtree.nodes.size()?1:0); // left subtree (if exists) is placed right after the root -> index=1
            transform(left_subtree.nodes.cbegin(), // source from
                      left_subtree.nodes.cend(),   // source till
                      tree.nodes.begin() + 1, // destination
                      [] (Tree::Node node) {
                          // increment indecies only for existing children, left 0 for non-existing
                          if( node.left_child ) node.left_child  += 1;
                          if( node.right_child) node.right_child += 1;
                          return node;
                      }
            );
            unsigned int offset = left_subtree.nodes.size();
            local_root.right_child = (offset+right_subtree.nodes.size() ? 1 + offset : 0); // right subtree is placed after the left one
            transform(right_subtree.nodes.cbegin(), // source from
                      right_subtree.nodes.cend(),   // source till
                      tree.nodes.begin() + 1 + offset, // destination
                      [offset] (Tree::Node node) {
                          // increment indecies only for existing children, left 0 for non-existing
                          if( node.left_child ) node.left_child  += 1 + offset;
                          if( node.right_child) node.right_child += 1 + offset;
                          return node;
                      }
            );
            return tree;
//        }
    }

    vector<Tree> ensemble;

public:
    double regress(const DataRow& row) const {
        double sum = 0.;
        for(const auto &tree : ensemble)
            sum += tree.predict(row).asFloating;
        return sum/ensemble.size();
    }

    int   classify(const DataRow& row) const { return 0; }

    void train(const DataFrame& df, const vector<unsigned int>& predictorsIdx, unsigned int responseIdx) {
        if( df.nrow() < 1 ) return ;
        rState.seed(0);
        const int nTrees = 10;
        for(unsigned int t=0; t<nTrees; t++){
            SplitVars vars( generateRandomSplitVars( df.getSchema(), predictorsIdx, floor(predictorsIdx.size()>15?predictorsIdx.size()/3:5) ) );//(unsigned int)sqrt(predictorsIdx.size()) ) );
//for(auto s : vars) cout << "s.first = "<<s.first << " s.second = "<< s.second << endl;
//            future<Tree> ft = async(std::launch::async, pickStrongestCuts, df, responseIdx, vars, sample(df.nrow(),df.nrow()*0.5));
            Tree tree = findBestSplits(df, responseIdx, vars, sample(df.nrow(),df.nrow()*0.5));
            ensemble.push_back( move(tree) );
        }
        
    }

    RandomForest(void){}

//    load/save
};

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
    ifstream input("one.csv");
    setSeparators(input);
    readHeader(input,3);
    typedef tuple<string,float,float> Format;
    DataFrame df;
    Format tmp;
    for(unsigned int row=0; read_tuple(input,tmp); row++){
        tuple<float,float> r12 = make_tuple(
                get<1>(tmp), get<2>(tmp)
        );
        df.rbind( DataRow(r12) );
    }
    return df;
}

DataFrame read2(void){
    ifstream input("../trigger/pt/SingleMu_Pt1To1000_FlatRandomOneOverPt.csv");
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
    DataFrame df;
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
    return df;
}


int main(void){
    RandomForest rf;

    DataFrame df( read2() );
    vector<unsigned int> predictorsIdx = {0};//,1,2,3,4,5};
    rf.train(df,predictorsIdx,6);

//    DataFrame df( read1() );
//    vector<unsigned int> predictorsIdx = {0,1};
//    rf.train(df,predictorsIdx,2);

//    rf.ensemble[0].save(cout);

    double bias = 0, var = 0;
    long cnt = 0;
    for(unsigned int row = 0; row>=0 && row < df.nrow(); row++,cnt++){
        double prediction = rf.regress( df[row] );
        double truth      = df[row][6].asFloating;
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
