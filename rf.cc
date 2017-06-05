#include <iostream>
#include <tuple>
#include <vector>
#include <iterator>
#include <algorithm>
#include <string>
#include <random>
#include <unordered_set>
using namespace std;

// g++ -Wall -std=c++11 -o rf rf.cc

// No better solution for the type-obsessed languages but to create a type
//  that can play for the both sides
struct Variable {
    enum Type { Unknown=0, Categorical=1, Continuous=2 };
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
    vector<int> schema; // 1 - continuous, >=2 - number of levels in categorical
    vector<DataRow> rows;

public:

    template<typename T>
    bool cbind(const vector<T> &col) {
        // check if number of rows matchs number of elements in column
        if( col.size() != rows.size() && rows.size() != 0 )
            return false;
        // in case the DataFrame is empty initialize it with the column
        if( rows.size() == 0 ) rows.resize( col.size() );
        // the two options: categorical/integral and floating/continuous
        if( std::is_integral<T>::value ){
            // deduce number of levels automatically
            unordered_set<long long> unique;
            for(unsigned i=0; i<col.size(); ++i){
                rows[i].data.push_back( Variable((long long) col[i]) );
                unique.insert((long long) col[i]);
            }
            // store number of found levels
            schema.push_back( unique.size() );
        } else {
            for(unsigned i=0; i<col.size(); ++i)
                rows[i].data.push_back( Variable((double)    col[i]) );
            // mark the column as continuous
            schema.push_back(1);
        }
        return true;
    }

    bool rbind(const DataRow &row) {
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
            // make sure we preserve the schema
            if( !equal(row.data.cbegin(), row.data.cend(), schema.cbegin(),
                     [](const Variable& var, int type){
                         return (var.type == Variable::Categorical && type >= 2) ||
                                (var.type == Variable::Continuous  && type == 1) ;
                     }
                 )
            ) return false;
        }
        rows.push_back(row);
        return true;
    }

    DataRow& operator[](unsigned int i) { return rows[i]; }

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
        Node(void) : value(), position(0), left_child(0), right_child(0){}
    };
    friend class Model;
    vector<Node> tree;

public:
    Variable traverse(const DataRow& row, const Node& root) const {
        // is it a leaf/terminal_node?
        if( root.left_child == 0 || root.right_child == 0 )
            return root.value;

        if( root.value.type == Variable::Continuous ){
            if( root.value.asFloating < row[root.position].asFloating )
                return traverse(row,tree[root.left_child]);
            else
                return traverse(row,tree[root.right_child]);
        }
        if( root.value.type == Variable::Categorical ){
            // only binary-level categories are managed
            if( root.value.asIntegral == row[root.position].asIntegral )
                return traverse(row,tree[root.left_child]);
            else
                return traverse(row,tree[root.right_child]);
        }
        // the root is neither Continuous nor Categorical -> error
        return Variable();
    }
    Variable predict(const DataRow& row) const {
        // is tree initialized? if not return default Variable as a sign of error
        if( tree.size() == 0 ) return Variable(); 
        // is root node initialized?
        if( tree[0].value.type == Variable::Unknown ) return Variable(); 
        // all looks good
        return traverse(row,tree[0]);
    }

//    load/save
};

class RandomForest {
private:
    unsigned int nInputs;

    std::default_random_engine rState;

    void preProcessDF(void){}

    unordered_set<unsigned int> generateRandomSplits(unsigned int mtry){
        unordered_set<unsigned int> retval;
        std::default_random_engine dre(rState);
        uniform_int_distribution<> uid(0, nInputs);
        generate_n( inserter(retval,retval.begin()), mtry, [&uid,&dre](void){ return uid(dre); } );
        return retval; // note, the retval is not always mtry-sized!
    }

    void sample(void){

    }

//    purity/gini/entrophy/rms

    vector<Tree> ensemble;

public:
    double regress(const DataRow& row) const { return 0; }
    int   classify(const DataRow& row) const { return 0; }
    void train(const DataFrame& df) {

    }

    RandomForest(void) : nInputs(0) {}

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

template <typename... Args>
double lossFunction(const std::tuple<Args...> p, const std::tuple<Args...> t) noexcept {
    return 0;
}

int main(void){
    // A categorical (potentially ordered) predictor type
    enum class cat1 : char { type1, type2, type3 };

    vector< tuple<cat1,float> > predictors = {
        make_tuple(cat1::type1,1.0),
        make_tuple(cat1::type2,2.0),
        make_tuple(cat1::type3,3.0)
    };

    vector< tuple<int> > targets = {
        make_tuple(1),
        make_tuple(2),
        make_tuple(3)
    };

//    RandomForests rf;

//    DataRow row(predictors[1]);

//    DataFrame (predictors,targets)
    DataRow row( make_tuple(1.1,1,true) );
    cout << row << endl;

    vector<int> col = {1,2,3,4};
//    DataFrame df;
//    df.cbind(col);
//    cout << df << endl;

    cout << oneHOTencode(col) << endl;

//    rf.train(predictors,targets,lossFunction);
//    fr.trees();

    return 0;
}
