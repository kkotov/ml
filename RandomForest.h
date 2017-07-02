#ifndef RandomForest_h
#define RandomForest_h

#include <map>
#include <list>
#include <vector>
#include <iterator>
#include <algorithm>
#include <functional>
#include <utility>
#include <random>
#include <limits>
#include <memory>
#include <unordered_map>

#include "DataFrame.h"
#include "Tree.h"

class RandomForest {
private:
public:
    typedef std::list<std::pair<unsigned int,unsigned int>> SplitVars; // variable index and level (if categorical)

    std::default_random_engine rState;

    SplitVars generateRandomSplitVars(const std::vector<std::vector<long>>& schema,
                                      const std::vector<unsigned int>& predictorsIdx,
                                      unsigned int mtry){
        SplitVars vars;
        std::default_random_engine dre(rState);
        std::uniform_int_distribution<unsigned int> uid(0, predictorsIdx.size()-1), uid_l;
        std::generate_n( back_inserter(vars),
                         mtry,
                         [&uid,&uid_l,&dre,&schema,&predictorsIdx](void){
                             unsigned int idx = predictorsIdx[ uid(dre) ];
                             unsigned int level = (schema[idx].size()>0 ? uid_l(dre)%schema[idx].size() : 0);
                            return std::pair<unsigned int,unsigned int>(idx,level);
                    }
        );
        return vars;
    }

    std::vector<unsigned int> sample(unsigned int nTotal,
                                     unsigned int nSampled,
                                     bool replace = false){
        // definitely, there is room for improvement below
        std::vector<unsigned int> retval(nTotal);
        if( !replace ){
            std::iota(retval.begin(),retval.end(),0);
            std::shuffle(retval.begin(),retval.end(),rState);
        } else {
            std::default_random_engine dre(rState);
            std::uniform_int_distribution<> uid(0, nTotal);
            std::generate_n( retval.begin(),
                             (nSampled < nTotal ? nSampled : nTotal),
                             [&uid, &dre](void){ return uid(dre); }
            );
        }
        return std::vector<unsigned int>(retval.begin(),
                                         retval.begin() +
                                         (nSampled < nTotal ? nSampled : nTotal)
               );
    }

    // thread-safe implementation of CART with gini/entropy/rms purity metrices
    Tree* findBestSplits(const DataFrame& df,
                         unsigned int responseIdx,
                         const SplitVars& vars,
                         const std::vector<unsigned int>& subset = {}
    ){
        Tree *tree = new Tree();

        // safety: nothing to split on? - return an empty tree
        if( vars.empty() ) return tree;

        size_t size = subset.size();
        tree->set_size = size;

// criterion to stop growing tree
#define MIN_ENTRIES 5

        if( df.getLevels(responseIdx).size() == 0 ){
            // response is Variable::Continuous
            // caclulate general regression characteristics on training events:
            //  RSS, sum, and sum of squares of responces 
            tree->rss  = 0;
            tree->sum  = 0;
            tree->sum2 = 0;
            for(unsigned int i=0; i<size; ++i){
                unsigned int row = subset[i];
                tree->sum  += df[row][responseIdx].asFloating;
                tree->sum2 += df[row][responseIdx].asFloating * df[row][responseIdx].asFloating;
            }
            tree->rss  = tree->sum2 - tree->sum * tree->sum / size;
            tree->metric = tree->rss;
        } else {
            // response is Variable::Categorical
            // caclulate general classification characteristics on training events:
            //  gini and cross-entropy
            tree->gini = 0;
            tree->crossEntropy = 0;
            for(unsigned int i=0; i<size; ++i){
                unsigned int row = subset[i];
                tree->levelCounts[ df[row][responseIdx].asIntegral ]++;
            }
            double maxProb = 0;
            for(auto c : tree->levelCounts){
                double p = double(c.second) / size;
                tree->gini         += p * (1 - p);
                tree->crossEntropy -= p * log(p);
                if( maxProb < p ){
                    maxProb = p;
                    tree->majorityVote = c.first;
                }
            }
            tree->metric = tree->gini;
        }

        // do not grow tree beyond MIN_ENTRIES or less 
        if( size <= MIN_ENTRIES ){
            if( df.getLevels(responseIdx).size() == 0 )
                tree->nodes.push_back( Tree::Node( double(tree->sum/size) ) );
            else
                tree->nodes.push_back( Tree::Node( long(tree->majorityVote) ) );
            tree->tree_size = 1; // leaf
            return tree;
        }

        // find best split
        size_t    bestSplitVar = 0;
        long long bestSplitPoint = 0; // integral split points to a row for continuous vars and to a level for factors
        double    bestSplitMetric = std::numeric_limits<double>::max();

        // functional form of Eq 9.13 on p.307 of ESLII (*_l - left to the cut, *_r - right)
        std::function<double(double,double,size_t,double,double,size_t)> rssMetric =
            [](double sum_l, double sum2_l, size_t size_l, double sum_r, double sum2_r, size_t size_r){
                return (size_l ? (sum2_l - sum_l * sum_l / size_l) : 0) +
                       (size_r ? (sum2_r - sum_r * sum_r / size_r) : 0);
            };

        // functional form for gini index given in Eq 9.17 on p.309 of ESLII
        std::function<double(const std::unordered_map<long long, size_t>&, size_t,
                             const std::unordered_map<long long, size_t>&, size_t)>
            giniMetric =
                [](const std::unordered_map<long long, size_t>& cnt_l, size_t size_l,
                   const std::unordered_map<long long, size_t>& cnt_r, size_t size_r){
                       size_t gini_l = 0;
                       for(auto c : cnt_l)
                           gini_l += c.second * (size_l - c.second);
                       size_t gini_r = 0;
                       for(auto c : cnt_r)
                           gini_r += c.second * (size_r - c.second);
                       return (size_l ? double(gini_l)/size_l/size_l : 0) +
                              (size_r ? double(gini_r)/size_r/size_r : 0);
                };

        // loop over split candidates, greedily select the best one
        for(std::pair<unsigned int,unsigned int> var : vars){

            long long bestSplitPointSoFar = 0;
            double    bestMetricSoFar = std::numeric_limits<double>::max();

            // continuous predictor?
            if( df.getLevels(var.first).size() == 0 ){

                // order training entries (well, through their indices) along the candidate variable for the split
                std::vector<unsigned int> indices(size);
                std::iota(indices.begin(),indices.end(),0);
                std::sort(indices.begin(),
                          indices.end(),
                          [&subset, &df, &var] (unsigned int i, unsigned int j) {
                              return df[ subset[i] ][var.first].asFloating < df[ subset[j] ][var.first].asFloating;
                          }
                );

                // regression for continuous case else classification for multilevel response
                if( df.getLevels(responseIdx).size() == 0 ){
                    // start with all points being on one (right) side of the split
                    double sum_r = tree->sum, sum2_r = tree->sum2, sum_l = 0, sum2_l = 0;
                    size_t size_l = 0, size_r = size;
                    bestMetricSoFar = rssMetric(sum_l,sum2_l,size_l,sum_r,sum2_r,size_r);
                    // and run over df subset sorted along current predictor
                    for(unsigned int index : indices){
                        unsigned int row = subset[index];
                        // advancing the split - moving a point from right to left of the split
                        sum_r  -= df[row][responseIdx].asFloating;
                        sum2_r -= df[row][responseIdx].asFloating * df[row][responseIdx].asFloating;
                        sum_l  += df[row][responseIdx].asFloating;
                        sum2_l += df[row][responseIdx].asFloating * df[row][responseIdx].asFloating;
                        size_r--;
                        size_l++;
                        double newMetric = rssMetric(sum_l,sum2_l,size_l,sum_r,sum2_r,size_r);
                        // check for the best split
                        if( newMetric < bestMetricSoFar ){
                            bestMetricSoFar     = newMetric;
                            bestSplitPointSoFar = row;
                        }
                    }
                } else { // multilevel response
                    // start with all points being on one (right) side of the split
                    std::unordered_map<long long, size_t> counts_r( tree->levelCounts ), counts_l;
                    size_t size_l = 0, size_r = size;
                    bestMetricSoFar = giniMetric(counts_l,size_l,counts_r,size_r);
                    for(unsigned int index : indices){
                        unsigned int row = subset[index];
                        // advancing the split - moving a point from right to left of the split
                        counts_r[ df[row][responseIdx].asIntegral ]--;
                        counts_l[ df[row][responseIdx].asIntegral ]++;
                        size_r--;
                        size_l++;
                        double newMetric = giniMetric(counts_l,size_l,counts_r,size_r);
                        if( newMetric < bestMetricSoFar ){
                            bestMetricSoFar     = newMetric;
                            bestSplitPointSoFar = row;
                        }
                    }
                }

            } else {
                // categorical predictor
                long level = df.getLevels(var.first)[var.second];
                bestSplitPointSoFar = level;
                // regression for continuous case else classification for multilevel response
                if( df.getLevels(responseIdx).size() == 0 ){
                    double sum_match = 0, sum2_match = 0;
                    size_t size_match = 0;
                    for(unsigned int i=0; i<size; ++i){
                        unsigned int row = subset[i];
                        if( df[row][var.first].asIntegral == level ){
                            sum_match  += df[row][responseIdx].asFloating;
                            sum2_match += df[row][responseIdx].asFloating * df[row][responseIdx].asFloating;
                            size_match++;
                        }
                    }
                    bestMetricSoFar =
                        rssMetric(sum_match, sum2_match, size_match,
                                  tree->sum-sum_match, tree->sum2-sum2_match, size-size_match
                        );
                } else { // multilevel response
                    std::unordered_map<long long, size_t> counts_match, counts_mismatch;
                    size_t size_match = 0;
                    for(unsigned int i=0; i<size; ++i){
                        unsigned int row = subset[i];
                        if( df[row][var.first].asIntegral == level ){
                            counts_match[ df[row][responseIdx].asIntegral ]++;
                            size_match++;
                        } else
                            counts_mismatch[ df[row][responseIdx].asIntegral ]++;
                    }
                    bestMetricSoFar =
                        giniMetric(counts_match,size_match,counts_mismatch,size-size_match);
                }
            }

            if( bestMetricSoFar < bestSplitMetric ){
                bestSplitVar    = var.first;
                bestSplitPoint  = bestSplitPointSoFar;
                bestSplitMetric = bestMetricSoFar;
            }
        }

        std::vector<unsigned int> left_subset, right_subset;
        for(unsigned int i : subset)
            switch(  df[i][bestSplitVar].type ){
                case Variable::Continuous:
                    if( df[i][bestSplitVar].asFloating < df[bestSplitPoint][bestSplitVar].asFloating )
                        left_subset.push_back(i);
                    else
                        right_subset.push_back(i);
                break ;
                case Variable::Categorical:
                    if( df[i][bestSplitVar].asIntegral == bestSplitPoint )
                        left_subset.push_back(i);
                    else
                        right_subset.push_back(i);
                    break ;
                default : return new Tree(); break;
        }

        // Continue growing tree until any of the subsets are smaller the MIN_ENTRIES
        if( left_subset.size() > MIN_ENTRIES && right_subset.size() > MIN_ENTRIES ){

            // another good place to use the threads
            Tree *left_subtree  = findBestSplits(df, responseIdx, vars, left_subset);
            Tree *right_subtree = findBestSplits(df, responseIdx, vars, right_subset);

            left_subtree->parent  = tree;
            right_subtree->parent = tree;
            tree->left_subtree  = left_subtree;
            tree->right_subtree = right_subtree;
            tree->nodes.resize(1);
            tree->tree_size = 1 + left_subtree->tree_size + right_subtree->tree_size;

            // copy the local root node
            Tree::Node& local_root = tree->nodes[0];
            local_root.position = bestSplitVar;
            if( df.getLevels(bestSplitVar).size() == 0 ){
                local_root.value.type = Variable::Continuous;
                local_root.value.asFloating = df[bestSplitPoint][bestSplitVar].asFloating;
            } else {
                local_root.value.type = Variable::Categorical;
                local_root.value.asIntegral = bestSplitPoint;
            }

        } else {
            if( df.getLevels(responseIdx).size() == 0 )
                tree->nodes.push_back( Tree::Node( double(tree->sum/size) ) );
            else
                tree->nodes.push_back( Tree::Node( long(tree->majorityVote) ) );
            tree->tree_size = 1; // leaf
        }
#undef MIN_ENTRIES

        return tree;
    }

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

    void train(const DataFrame& df, const std::vector<unsigned int>& predictorsIdx, unsigned int responseIdx) {
        if( df.nrow() < 1 ) return ;
        // reproducibility
        rState.seed(0);
        const int nTrees = 1;
        for(unsigned int t=0; t<nTrees; t++){
            SplitVars vars(
                generateRandomSplitVars(
                    df.getSchema(),
                    predictorsIdx,
                    ( df.getLevels(responseIdx).size() == 0 ?
                        std::floor(predictorsIdx.size()>15 ? predictorsIdx.size()/3 : 5) :
                        std::floor( sqrt(predictorsIdx.size()) )
                    )
                )
            );
//for(auto s : vars) std::cout << "s.first = "<<s.first << " s.second = "<< s.second << std::endl;

            // cross-validate models over nFolds for evaluating the best model complexity
            const size_t nFolds = 10;
            std::vector<double> alphas[nFolds]; // series of hyper-parameter for each fold
            std::vector<std::shared_ptr<Tree>> models[nFolds]; // series of corresponding models for each fold
            std::vector<double> outOfSampleError[nFolds]; // OOB error estimate
            std::vector<unsigned int> shuffled = sample(df.nrow(),df.nrow());
            for(size_t fold=0; fold<nFolds; fold++){
                // partition into training and validation sets
                std::vector<unsigned int> trainSet, validSet;
                trainSet.reserve( shuffled.size() );
                validSet.reserve( shuffled.size() );
                std::vector<unsigned int>::const_iterator begin = shuffled.cbegin() + (shuffled.size()*fold)/nFolds;
                std::vector<unsigned int>::const_iterator   end = shuffled.cbegin() + (shuffled.size()*(fold+1))/nFolds;
                std::copy(shuffled.cbegin(), begin,           std::back_inserter(trainSet));
                std::copy(end,               shuffled.cend(), std::back_inserter(trainSet));
                std::copy(begin,             end,             std::back_inserter(validSet));
                // get the tree
                Tree *tree = findBestSplits(df, responseIdx, vars, trainSet);
                // and prune it
                std::map<double,std::shared_ptr<Tree>> m = tree->prune();
                alphas[fold].reserve( m.size() );
                models[fold].reserve( m.size() );
                outOfSampleError[fold].reserve( m.size() );
                // remember
                //  hyper-parameters
                std::transform(m.cbegin(),
                               m.cend(),
                               std::back_inserter(alphas[fold]),
                               [](std::pair<double,std::shared_ptr<Tree>> a){
                                   return a.first;
                               }
                );
                //  corresponding models
                std::transform(m.cbegin(),
                               m.cend(),
                               std::back_inserter(models[fold]),
                               [](std::pair<double,std::shared_ptr<Tree>> a){ 
                                   return a.second;
                               }
                );
                //  and their out-of-sample errors
                std::transform(m.cbegin(),
                               m.cend(),
                               std::back_inserter(outOfSampleError[fold]),
                               [&df, &responseIdx, &validSet](std::pair<double,std::shared_ptr<Tree>> a){
                                   return a.second->evaluateMetric(df, responseIdx, validSet);
                               }
                );
            }

            // among different model complexities pick one
            //  "...to minimize the average error." (slide #20 of 08-trees-handout.pdf of StatLearn)
            std::map<double,double> averageError;
            // run over alpha hyper-parameter that controls model complexity
            size_t pos[nFolds] = {};
            while(1){
                // find fold holding the next smallest alpha
                double nextAlpha = std::numeric_limits<double>::max();
                int nextFold = -1;
                for(size_t fold=0; fold<nFolds; fold++){
                    if( pos[fold] == alphas[fold].size() - 1 ) continue;
                    if( nextAlpha > alphas[fold][ pos[fold] + 1 ] ){
                        nextAlpha = alphas[fold][ pos[fold] + 1 ]; 
                        nextFold  = fold;
                    }
                }
                // no more alphas remain
                if( nextFold < 0 ) break;

                pos[nextFold]++;

                double aveErr = 0;
                for(size_t fold=0; fold<nFolds; fold++)
                    aveErr += outOfSampleError[fold][ pos[fold] ];
                aveErr /= nFolds;

                averageError.insert(std::make_pair(nextAlpha,aveErr));
            }

            std::map<double,double>::const_iterator it =
                std::min_element(averageError.begin(),
                                 averageError.end(),
                                 [](std::pair<double,double> a, std::pair<double,double> b){
                                     return a.second < b.second;
                                 }
                );

            double bestAlpha = it->first;
std::cout << "bestAlpha=" << bestAlpha << std::endl;

            Tree *tree = findBestSplits(df, responseIdx, vars, shuffled);
            std::map<double,std::shared_ptr<Tree>> m = tree->prune();
            auto model = m.lower_bound(bestAlpha);

//model->second->save(std::cout);
            ensemble.push_back( std::move(*(model->second)) );
        }
    }

    RandomForest(void){}

//    load/save
};

#endif
