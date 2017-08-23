#ifndef RandomForest_h
#define RandomForest_h

#include <unordered_map>
#include <algorithm>
#include <vector>

#include "DataFrame.h"
#include "Tree.h"
#include "TreeTrainer.h"

class RandomForest {
private:
    std::vector<Tree> ensemble;

public:
    double regress(const DataRow& row) const {
        double sum = 0.;
        for(const auto &tree : ensemble){
            Variable v = tree.predict(row);
            if( v.type != Variable::Continuous )
                return 0; // maybe throw?
            sum += v.asFloating;
        }
        return sum/ensemble.size();
    }

    int classify(const DataRow& row) const {
        std::unordered_map<long long, size_t> votes;
        for(const auto &tree : ensemble){
            Variable v = tree.predict(row);
            if( v.type != Variable::Categorical )
                return 0; // maybe throw?
            votes[ v.asIntegral ]++;
        }
        std::unordered_map<long long, size_t>::const_iterator it =
            std::max_element(votes.cbegin(),
                             votes.cend(),
                             [](std::pair<long long, size_t> a, std::pair<long long, size_t> b){
                                 return a.second < b.second;
                             }
            );
        return it->first;
    }

    void train(const DataFrame& df, const std::vector<unsigned int>& predictorsIdx, unsigned int responseIdx, size_t nTrees = 100) {
        TreeTrainer tt;
        std::vector<std::shared_ptr<Tree>> treePtrs = tt.trainRandomForest(df, predictorsIdx, responseIdx, nTrees, 0);
        ensemble.resize( treePtrs.size() );
        for(size_t i=0; i<treePtrs.size(); i++)
            ensemble[i] = *treePtrs[i];
    }

    bool load(std::istream& input){
        bool status = true;
        std::string tmp; 
        unsigned int nTrees = 0;

        input >> tmp >> nTrees;
        ensemble.resize(nTrees);

        for(unsigned int n=0, i=0; n<nTrees && status; n++){
            input >> tmp >> i;
            if( i != n ) return false;
            status &= ensemble[n].load(input);
        }

        return status;
    }

    bool save(std::ostream& output) const {
        bool status = true;
        output << "nTrees: " << ensemble.size() << std::endl;
        for(unsigned int n=0; n<ensemble.size(); n++){
            output << "tree: " << n << std::endl;
            status &= ensemble[n].save(output);
        }
        return status;
    }

};

#endif
