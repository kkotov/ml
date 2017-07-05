#ifndef DataFrame_h
#define DataFrame_h

#include <tuple>
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <unordered_set>

// No better solution for the type-obsessed languages but to create a type
//  that can play for the both sides
struct Variable {
    enum Type { Unknown=0, Categorical=1, Continuous=2 }; // Categorical is always considered unordered below
    Type type;
    union {
        long long asIntegral;
        double    asFloating;
    };
    friend std::ostream& operator<< (std::ostream&, const Variable&);
    std::ostream& operator<< (std::ostream& out) const {
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

inline std::ostream& operator<< (std::ostream& out, const Variable& var) { return var.operator<<(out); }

// The DataRow abstraction is meant to be an interface between templated and
//  non-templated worlds. Although the whole RandomForests framework could
//  have been made templated, this brings an unnecessary generalization and
//  blows the code out of proportion, while in fact we only deal with two types
//  of variables: categorical (integral) and non-categorical (floating point)
class DataRow {
private:
    // all the elements of the row are stored in this vector
    std::vector<Variable> data;

    // helper: store element with index IDX of tuple with MAX elements
    template <int IDX, int MAX, typename... Args>
    struct STORE_TUPLE {
        static void store(std::vector<Variable>& d, const std::tuple<Args...>& t) {
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
        static void store(std::vector<Variable>& d, const std::tuple<Args...>& t) {}
    };

    friend class DataFrame;

public:

    // interfacing DataRow with tuple
    template <typename... Args>
    DataRow& operator=(const std::tuple<Args...> &t) noexcept {
        data.resize(0);
        STORE_TUPLE<0,sizeof...(Args),Args...>::store(data,t);
        return *this;
    }

    template <typename... Args>
    DataRow(const std::tuple<Args...> &t) noexcept {
        STORE_TUPLE<0,sizeof...(Args),Args...>::store(data,t);
    }

    // subsetting
          Variable& operator[](unsigned int i)       { return data[i]; }
    const Variable& operator[](unsigned int i) const { return data[i]; }

    std::ostream& operator<< (std::ostream& out) const {
        std::copy(data.cbegin(),data.cend(),std::ostream_iterator<Variable>(out," "));
        return out;
    }

    DataRow(void){}
    // copy, and move c-tors will be generated by the compiler
};

inline std::ostream& operator<< (std::ostream& out, const DataRow& dr) { return dr.operator<<(out); }

// abstraction for grouping DataRows together
class DataFrame {
private:
    std::vector<std::vector<long>> schema; // levels (empty for continuous columns)
    std::vector<DataRow> rows;

public:
    const std::vector<std::vector<long>>& getSchema(void) const { return schema; }
    const std::vector<long>& getLevels(unsigned int colidx) const { return schema[colidx]; }

    unsigned int nrow(void) const { return rows.size(); }
    unsigned int ncol(void) const { return schema.size(); }

    template<typename T>
    bool cbind(const std::vector<T> &col, std::vector<long> levels={}) {
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
            std::unordered_set<long> unique;
            std::copy(col.cbegin(), col.cend(), std::inserter(unique,unique.begin()));
            // store found or provided levels
            if( levels.size() > unique.size() )
                schema.push_back( levels );
            else {
                std::vector<long> tmp(unique.size());
                std::copy(unique.cbegin(), unique.cend(), tmp.begin());
                schema.push_back( tmp );
            }
        } else {
            for(unsigned i=0; i<col.size(); ++i)
                rows[i].data.emplace_back((double)col[i]);
            // mark the column as continuous
            schema.push_back(std::vector<long>());
        }
        return true;
    }

    bool rbind(const DataRow& row) {
        // check if number of elements in the row agrees with the expectation
        if( row.data.size() != schema.size() && schema.size() > 0 )
            return false;
        // check if we start fresh
        if( schema.size() == 0 ){
            // initialize the empty DataFrame with the row
            rows.push_back(row);
            std::transform(row.data.cbegin(), row.data.cend(), std::back_inserter(schema),
                [](const Variable& var){ // consider just two levels, recalculate later
                    return (var.type == Variable::Categorical ? std::vector<long>(2) : std::vector<long>());
                }
            );
        } else {
            // make sure we preserve the schema but do nothing about number of levels yet
            if( !std::equal(row.data.cbegin(), row.data.cend(), schema.cbegin(),
                     [](const Variable& var, const std::vector<long>& levels){
                         return (var.type == Variable::Categorical && levels.size() >= 1) ||
                                (var.type == Variable::Continuous  && levels.empty()) ;
                     }
                 )
            ) return false;
        }
        //rows.push_back( std::move(row) );
        rows.push_back(row);
        return true;
    }

    void countLevels(unsigned int colidx, std::vector<long> hint={}){
        // nothing to do for continuous case
        if( schema[colidx].empty() ) return;
        // deduce number of levels automatically
        std::unordered_set<long> unique;
        std::transform(rows.cbegin(), rows.cend(), std::inserter(unique,unique.begin()),
            [colidx](const DataRow& row){
                return row[colidx].asIntegral;
            }
        );
        // store found or provided levels
        if( hint.size() > unique.size() )
            schema[colidx] = hint;
        else {
            std::vector<long> tmp(unique.size());
            std::copy(unique.cbegin(), unique.cend(), tmp.begin());
            schema[colidx] = tmp;
        }
    }

    void countAllLevels(void){
        for(unsigned int colidx=0; colidx<schema.size(); colidx++)
            countLevels(colidx);
    }

          DataRow& operator[](unsigned int i)       { return rows[i]; }
    const DataRow& operator[](unsigned int i) const { return rows[i]; }

    std::ostream& print(std::ostream& out, int nrows=-1) const {
        std::copy( rows.cbegin(),
                   (nrows<0 ? rows.cend() : rows.cbegin()+nrows),
                   std::ostream_iterator<DataRow>(out,"\n")
        );
        return out;
    }
};

inline std::ostream& operator<<(std::ostream& out, const DataFrame& df){ return df.print(out); }

#endif
