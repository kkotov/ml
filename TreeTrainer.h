#ifndef TreeTrainer_h
#define TreeTrainer_h

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

// thread-safe stateless class encompassing static functions only
class TreeTrainer {
private:
    typedef std::vector<std::pair<unsigned int,unsigned int>> SplitVars; // variable index and level index (if categorical)

    static SplitVars generateRandomSplitVars(const std::vector<std::vector<long>>& schema,
                                             const std::vector<unsigned int>& predictorsIdx,
                                             unsigned int mtry,
                                             std::default_random_engine& rState)
    {
        // sample mtry variables without replacement from all the variables
        std::vector<unsigned int> indices( predictorsIdx.size() );
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rState);
        indices.resize(mtry);

        // turn multilevel categorical predictors into dummy variables
        SplitVars oneHOTencoded;

        for(unsigned int i : indices){
            unsigned int idx = predictorsIdx[i];
            unsigned int nLevels = schema[idx].size();
            if( nLevels > 0 )
                for(unsigned int levelIdx = 0; levelIdx < nLevels; ++levelIdx)
                    oneHOTencoded.push_back(std::make_pair(idx, levelIdx));
            else
                oneHOTencoded.push_back(std::make_pair(idx,0));
        }

        return oneHOTencoded;
    }

    static std::vector<unsigned int> sample(unsigned int nTotal,
                                            unsigned int nSampled,
                                            std::default_random_engine &rState, // feed the current state back to the call context
                                            bool replace = false)
    {
        // definitely, there is room for improvement below
        std::vector<unsigned int> retval(nTotal);
        if( !replace ){
            std::iota(retval.begin(),retval.end(),0);
            std::shuffle(retval.begin(),retval.end(),rState);
        } else {
            std::uniform_int_distribution<> uid(0, nTotal-1); // exclude max value
            std::generate_n( retval.begin(),
                             (nSampled < nTotal ? nSampled : nTotal),
                             [&uid, &rState](void){ return uid(rState); }
            );
        }
        return std::vector<unsigned int>(retval.begin(),
                                         retval.begin() +
                                         (nSampled < nTotal ? nSampled : nTotal)
               );
    }

    // thread-safe implementation of CART with gini/entropy/rms purity metrics
    static Tree* findBestSplits(const DataFrame& df,
                                unsigned int responseIdx,
                                const std::vector<unsigned int>& predictorsIdx,
                                const std::vector<unsigned int>& subset,
                                std::default_random_engine &rState, // feed the current state back to the call context
                                unsigned int mtry, // if !=0, this is RandomForest
                                size_t MIN_ENTRIES = 5) // criterion to stop growing tree
                                // an optional last argument here defines a number
                                // of events in terminal nodes and can help to
                                // speed up the tree-growing process for big datasets
                                // (at expense of performance, of course)
    {
        Tree *tree = new Tree();

        // safety: nothing to split on? - return an empty tree
        if( predictorsIdx.empty() ) return tree;

        size_t size = subset.size();
        tree->set_size = size;

        if( df.getLevels(responseIdx).size() == 0 ){
            // response is Variable::Continuous
            // calculate general regression characteristics on training events:
            //  RSS, sum, and sum of squares of responses 
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
            // calculate general classification characteristics on training events:
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
        long long bestSplitPoint = -1; // integral split points to a row for continuous predictors and to a level's index for factors
        long long nextSplitPoint = -1; // row following the best split row
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
                       return (size_l ? double(gini_l)/size_l : 0) +
                              (size_r ? double(gini_r)/size_r : 0);
                };

        // the only Random-Forest specific step here: draw a random subset of predictors if mtry!=0
        SplitVars vars;
        if( mtry ){
            vars = generateRandomSplitVars(
                      df.getSchema(),
                      predictorsIdx,
                      mtry,
                      rState
                   );
        } else {
            for(unsigned int idx : predictorsIdx){
                if( df.getLevels(idx).size() == 0 )
                    vars.push_back(std::make_pair(idx,0));
                else {
                    // one-HOT encoding
                    for(unsigned int levelIdx = 0; levelIdx < df.getLevels(idx).size(); ++levelIdx)
                        vars.push_back(std::make_pair(idx,levelIdx));
                }
            }
        }

        // loop over split candidates, greedily select the best one
        for(std::pair<unsigned int,unsigned int> var : vars){

            long long bestSplitPointSoFar = -1; // that'll point to a row or left -1 if no split was found
            long long nextSplitPointSoFar = -1; // the same
            double    bestMetricSoFar = std::numeric_limits<double>::max();

            // continuous predictor?
            if( df.getLevels(var.first).size() == 0 ){

                // order training entries (well, through their indices) along the candidate predictor for the split
                std::vector<unsigned int> indices(size);
                std::iota(indices.begin(),indices.end(),0);
                std::sort(indices.begin(),
                          indices.end(),
                          [&subset, &df, &var] (unsigned int i, unsigned int j) {
                              return df[ subset[i] ][var.first].asFloating < df[ subset[j] ][var.first].asFloating;
                          }
                );
                // Note: with just a bit of modification of the code below the NlogN complexity from
                //  sorting above could have been avoided leaving us with much faster linear performance

                // current predictor is one repeated value, no split can be applied in such case
                //  just move on to the next predictor
                if( std::abs( df[ subset[indices[0]]      ][var.first].asFloating -
                              df[ subset[indices[size-1]] ][var.first].asFloating
                            ) <= std::numeric_limits<float>::min() )
                    continue;

                // regression for continuous case else classification for multilevel response
                if( df.getLevels(responseIdx).size() == 0 ){
                    // start with all points being on one (right) side of the split
                    double sum_r = tree->sum, sum2_r = tree->sum2, sum_l = 0, sum2_l = 0;
                    size_t size_l = 0, size_r = size;
                    // retarded values for the "look-ahead" loop below
                    int    prev_row = -1;
                    double prev_val = std::numeric_limits<double>::lowest();
                    double prev_metric = rssMetric(sum_l,sum2_l,size_l,sum_r,sum2_r,size_r);
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
                        double new_metric = rssMetric(sum_l,sum2_l,size_l,sum_r,sum2_r,size_r);
                        // check for the best split
                        //  caution here: repeating predictor's values cannot be ordered;
                        //  that means I either move around the split (from right to left) all of the values or none of them;
                        //  therefore, I see if current value advanced and update the split for the retarded row if needed
                        if( prev_val < df[row][var.first].asFloating && prev_metric < bestMetricSoFar ){
                            bestMetricSoFar     = prev_metric;
                            bestSplitPointSoFar = prev_row; // first time I'm here it is still -1
                            nextSplitPointSoFar = row;
                        }
                        prev_metric = new_metric;
                        prev_val = df[row][var.first].asFloating + std::numeric_limits<float>::min();
                        prev_row = row;
                    }
                    // because of metric property to to be the same on both ends I ignore the last row

                    // no best split was achieved which may happen only if response doesn't ever change
                    //  in such case no best split may be achieved with any predictor, just terminate the search
                    if( bestSplitPointSoFar < 0 ) break;

                } else { // multilevel response
                    // start with all points being on one (right) side of the split
                    std::unordered_map<long long, size_t> counts_r( tree->levelCounts ), counts_l;
                    size_t size_l = 0, size_r = size;
                    // retarded values for the "look-ahead" loop below
                    int    prev_row = -1;
                    double prev_val = std::numeric_limits<double>::lowest();
                    double prev_metric = giniMetric(counts_l,size_l,counts_r,size_r);
                    for(unsigned int index : indices){
                        unsigned int row = subset[index];
                        // advancing the split - moving a point from right to left of the split
                        counts_r[ df[row][responseIdx].asIntegral ]--;
                        counts_l[ df[row][responseIdx].asIntegral ]++;
                        size_r--;
                        size_l++;
                        double new_metric = giniMetric(counts_l,size_l,counts_r,size_r);
                        // check for the best split
                        //  same caution as in the previous block: repeating predictor's values cannot be ordered;
                        //  that means I either move around the split (from right to left) all of the values or none of them;
                        //  therefore, I see if current value advanced and update the split for the retarded row if needed
                        if( prev_val < df[row][var.first].asFloating && prev_metric < bestMetricSoFar ){
                            bestMetricSoFar     = prev_metric;
                            bestSplitPointSoFar = prev_row; // first time I'm here it is still -1
                            nextSplitPointSoFar = row;
                        }
                        prev_metric = new_metric;
                        prev_val = df[row][var.first].asFloating + std::numeric_limits<float>::min();
                        prev_row = row;
                    }
                    // no best split was achieved which may happen only if response doesn't ever change
                    //  in such case no best split may be achieved with any predictor, just terminate the search
                    if( bestSplitPointSoFar < 0 ) break;
                }

            } else {
                // categorical predictor
                bestSplitPointSoFar = var.second; // this is index of the level
                long level = df.getLevels(var.first)[bestSplitPointSoFar];
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
                    // if response doesn't change terminate the search
                    if( false ){
                    }
                    // if there are no other levels, no best split exist in this predictor
                    if( size_match == 0 || size_match == size ){
                        bestSplitPointSoFar = -1;
                        continue;
                    } else
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
                    // if there are no other levels, no best split exist in this predictor
                    if( size_match == 0 || size_match == size ){
                        bestSplitPointSoFar = -1;
                        continue;
                    } else 
                        bestMetricSoFar =
                            giniMetric(counts_match,size_match,counts_mismatch,size-size_match);
                }
            }

            if( bestMetricSoFar < bestSplitMetric ){
                bestSplitVar    = var.first;
                bestSplitPoint  = bestSplitPointSoFar;
                nextSplitPoint  = nextSplitPointSoFar;
                bestSplitMetric = bestMetricSoFar;
            }
        }

        std::vector<unsigned int> left_subset, right_subset;

        if( bestSplitPoint>=0 )
            for(unsigned int i : subset)
                switch(  df[i][bestSplitVar].type ){
                    case Variable::Continuous:
                        if( df[i][bestSplitVar].asFloating <= df[bestSplitPoint][bestSplitVar].asFloating )
                            left_subset.push_back(i);
                        else
                            right_subset.push_back(i);
                    break ;
                    case Variable::Categorical:
                        if( df[i][bestSplitVar].asIntegral == df.getLevels(bestSplitVar)[bestSplitPoint] )
                            left_subset.push_back(i);
                        else
                            right_subset.push_back(i);
                        break ;
                    default : return new Tree(); break;
            }

        // Continue growing tree until any of the subsets is not empty
        if( left_subset.size() > 0 && right_subset.size() > 0 ){

            // another good place to use the threads
            Tree *left_subtree  = findBestSplits(df, responseIdx, predictorsIdx, left_subset,  rState, mtry, MIN_ENTRIES);
            Tree *right_subtree = findBestSplits(df, responseIdx, predictorsIdx, right_subset, rState, mtry, MIN_ENTRIES);

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
                local_root.value.asFloating = ( df[bestSplitPoint][bestSplitVar].asFloating +
                                                df[nextSplitPoint][bestSplitVar].asFloating ) / 2.;
            } else {
                local_root.value.type = Variable::Categorical;
                local_root.value.asIntegral = df.getLevels(bestSplitVar)[bestSplitPoint];
            }

        } else {
            // turned out this is a pure node
            if( df.getLevels(responseIdx).size() == 0 )
                tree->nodes.push_back( Tree::Node( double(tree->sum/size) ) );
            else
                tree->nodes.push_back( Tree::Node( long(tree->majorityVote) ) );
            tree->tree_size = 1; // leaf
        }

        return tree;
    }

public:

    // train one simple Classification And Regression Tree
    static std::shared_ptr<Tree> trainCART(const DataFrame& df, const std::vector<unsigned int>& predictorsIdx, unsigned int responseIdx, long seed) {
        if( df.nrow() < 1 ) return std::shared_ptr<Tree>(); // maybe better to throw?

        std::default_random_engine rState(seed); // reproducibility: same seed results in identical trees

        // cross-validate models over nFolds for evaluating the best model complexity
        const size_t nFolds = 10;
        std::vector<double> alphas[nFolds]; // series of hyper-parameter for each fold
        std::vector<std::shared_ptr<Tree>> models[nFolds]; // series of corresponding models for each fold
        std::vector<double> outOfSampleError[nFolds]; // OOS error estimate
        std::vector<unsigned int> shuffled = sample(df.nrow(), df.nrow(), rState);
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
            Tree *tree = findBestSplits(df, responseIdx, predictorsIdx, trainSet, rState, 0);
            // and prune it
            std::map<double,std::shared_ptr<Tree>> m = tree->prune(); // guaranteed to create a new tree
            delete tree;
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

        Tree *tree = findBestSplits(df, responseIdx, predictorsIdx, shuffled, rState, 0);
        std::map<double,std::shared_ptr<Tree>> m = tree->prune();
        delete tree;
        auto model = m.lower_bound(bestAlpha);

        return model->second;
    }

    static std::shared_ptr<Tree> trainRFtree(const DataFrame& df,
                                             const std::vector<unsigned int>& predictorsIdx,
                                             unsigned int responseIdx,
                                             unsigned int seed = 0,
                                             unsigned int mtry = 0, // default value means auto-assign mtry
                                             double bootstrapSize = 0.632,
                                             unsigned int minNodeEntries = 5) {

        // reproducibility: same seed results in identical trees
        std::default_random_engine rState(seed);

        std::shared_ptr<Tree> tree( new Tree() );

        // auto-assign mtry if it is not provided
        if( mtry == 0 ){
            unsigned int nPredictors = predictorsIdx.size();
            mtry = ( df.getLevels(responseIdx).size() == 0 ? // continuous response?
                              std::floor( nPredictors>15 ? nPredictors/3 : (nPredictors > 5 ? 5 : nPredictors/2) ) :
                              std::floor( std::sqrt(nPredictors) )
                   );
        }

        // sample with replacement a fraction of data for training a new tree
        std::vector<unsigned int> s = sample(df.nrow(), df.nrow()*bootstrapSize, rState, true);
        Tree *tr = findBestSplits(df, responseIdx, predictorsIdx, s, rState, mtry, minNodeEntries);

        tree->nodes.reserve(tr->tree_size);
        tr->vectorize(tree->nodes);
        delete tr;

        return tree;
    }

};

#endif
