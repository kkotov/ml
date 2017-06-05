#include <iostream>
#include <tuple>
#include <vector>
#include <iterator>
#include <algorithm>
#include <string>
#include <random>
#include <unordered_set>
#include <valarray>
#include <iterator>
using namespace std;

// g++ -Wall -std=c++11 -o qwe valar.cc

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
    Variable& operator=(long long integral){ type = Categorical; asIntegral = integral; return *this; }
    Variable& operator=(double    floating){ type = Continuous;  asFloating = floating; return *this; }
    explicit Variable(long long integral){ type = Categorical; asIntegral = integral; }
    explicit Variable(double    floating){ type = Continuous;  asFloating = floating; }
    Variable(void){ type = Unknown; asIntegral = 0; }
};

ostream& operator<< (ostream& out, const Variable& var) { return var.operator<<(out); }

// The DataRow abstraction is meant to be an interface between templated and
//  non-templated worlds. Although the whole RandomForests framework could
//  have been made templated, this brings an unnecessary generalization and
//  blows the code out of proportion, while in fact we only deal with two types
//  of variables: categorical (integral) and non-categorical (floating point)
class DataRow {
private:
    // all the elements of the row are stored in this valarray (for convenient subsetting)
    valarray<Variable> data;

    // helper: store element with index IDX of tuple with MAX elements
    template <int IDX, int MAX, typename... Args>
    struct STORE_TUPLE {
        static void store(valarray<Variable>& d, const std::tuple<Args...>& t) {
            auto element = std::get<IDX>(t);
            if( std::is_integral<decltype(element)>::value )
                d[IDX] = (long long) element;
            else
                d[IDX] = (double   ) element;
            STORE_TUPLE<IDX+1,MAX,Args...>::store(d,t);
        }
    };

    // partial specialization to end the recursion
    template <int MAX, typename... Args>
    struct STORE_TUPLE<MAX,MAX,Args...> {
        static void store(valarray<Variable>& d, const std::tuple<Args...>& t) {}
    };

    friend class DataFrame;

public:

    // interfacing DataRow with tuple
    template <typename... Args>
    DataRow& operator=(const tuple<Args...> &t) noexcept {
        data.resize(sizeof...(Args));
        STORE_TUPLE<0,sizeof...(Args),Args...>::store(data,t);
        return *this;
    }

    template <typename... Args>
    DataRow(const tuple<Args...> &t) noexcept {
        data.resize(sizeof...(Args));
        STORE_TUPLE<0,sizeof...(Args),Args...>::store(data,t);
    }

    // subsetting
          Variable& operator[](unsigned int i)       { return data[i]; }
    const Variable& operator[](unsigned int i) const { return data[i]; }

    ostream& operator<< (ostream& out) const {
        copy(begin(data),end(data),ostream_iterator<Variable>(out," "));
        return out;
    }

    // a rather expensive linear time push_back
    void push_back(Variable var){
        valarray<Variable> new_data(data.size()+1);
        copy(begin(data), end(data), begin(new_data));
        new_data[data.size()] = var;
        data = move(new_data);
    }

    DataRow(void){}
    // copy, and move c-tors will be generated by the compiler
};

ostream& operator<< (ostream& out, const DataRow& dr) { return dr.operator<<(out); }

// abstraction for grouping DataRows together
class DataFrame {
private:
    vector<int> schema; // 1 - continuous, >=2 - number of levels in categorical
    valarray<DataRow> rows;

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
                rows[i].push_back( Variable((long long) col[i]) );
                unique.insert((long long) col[i]);
            }
            // store number of found levels
            schema.push_back( unique.size() );
        } else {
            for(unsigned i=0; i<col.size(); ++i)
                rows[i].push_back( Variable((double)    col[i]) );
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
///            rows.push_back(row);
            transform(begin(row.data), end(row.data), back_inserter(schema),
                [](const Variable& var){ return (var.type == Variable::Categorical ? 2 : 1) ; }
            );
        } else {
            // make sure we preserve the schema
            if( !equal(begin(row.data), end(row.data), schema.cbegin(),
                     [](const Variable& var, int type){
                         return (var.type == Variable::Categorical && type >= 2) ||
                                (var.type == Variable::Continuous  && type == 1) ;
                     }
                 )
            ) return false;
        }
///        rows.push_back(row);
        return true;
    }

    DataRow& operator[](unsigned int i) { return rows[i]; }

    ostream& print(ostream& out, int nrows=-1) const {
        copy( begin(rows),
              (nrows<0 ? end(rows) : begin(rows)+nrows),
              ostream_iterator<DataRow>(out,"\n")
        );
        return out;
    }
};

ostream& operator<<(ostream& out, const DataFrame& df){ return df.print(out); }
