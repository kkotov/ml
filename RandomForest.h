#ifndef RandomForest_h
#define RandomForest_h

#include <unordered_map>
#include <algorithm>
#include <vector>

#include <future>
#include <chrono>
#include <thread>

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

    void train(const DataFrame& df,
               const std::vector<unsigned int>& predictorsIdx,
               unsigned int responseIdx,
               size_t nTrees,
               std::ostream &log,
               unsigned int mtry = 0, // use auto-assign for mtry
               unsigned int seed = 0) {

        ensemble.resize(nTrees);

        const unsigned int maxThreads = std::thread::hardware_concurrency();

        std::future<std::shared_ptr<Tree>> results[maxThreads];
        unsigned int treeNumber[maxThreads];

        for(unsigned int tasksDone=0, freeThread=0, task=0; tasksDone<nTrees; ){
            // below I put together a rudimentary executor framework that limits a number of tasks running simultaneously

            // poll the status of the threads until any one of them frees up
            while( results[freeThread].valid() &&
                   results[freeThread].wait_for(std::chrono::seconds(0)) != std::future_status::ready )
            {
                // make sure the thread is/was running and force it to run if it is not
                if( results[freeThread].wait_for(std::chrono::seconds(0)) == std::future_status::deferred ){
                    ensemble[ treeNumber[freeThread] ] = *(results[freeThread].get());
                    tasksDone++;
                    break;
                }
                // try next thread
                freeThread = (freeThread + 1) % maxThreads;
                std::this_thread::yield();
            }

            // got here because the thread finished and holds result/exception
            if( results[freeThread].valid() ){
                ensemble[ treeNumber[freeThread] ] = *(results[freeThread].get());
                tasksDone++;
            }

            // dispatch a new task unless yet unfinished tasks wll already sum up to nTrees
            if( task < nTrees ){
                log << '\r' << "Training tree #" << (task+1) << "/" << nTrees << std::flush; // "\e[A"
                results[freeThread] = std::async(std::launch::async,
                                                 TreeTrainer::trainRFtree,
                                                 df,
                                                 predictorsIdx,
                                                 responseIdx,
                                                 seed + task + 1, // random sequences for 0 and 1 are identical -> offset by 1
                                                 mtry,
                                                 0.632, // bootstrapSize
                                                 5      // minNodeEntries
                                      );
                treeNumber[freeThread] = task;
                task++;
            } else freeThread = (freeThread + 1) % maxThreads;
        }

        log << std::endl;
    }

    bool load(std::istream& input){
        bool status = true;
        std::string tmp; 
        unsigned int nTrees = 0;

        input >> tmp >> nTrees;
        if( input.rdstate() & std::ios_base::failbit )
            return false;
        ensemble.resize(nTrees);

        for(unsigned int n=0, i=0; n<nTrees && status; n++){
            input >> tmp >> i;
            if( i != n || input.rdstate() & std::ios_base::failbit )
                return false;
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
