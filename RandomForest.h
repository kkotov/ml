#ifndef RandomForest_h
#define RandomForest_h

#include <list>
#include <queue>
#include <vector>
#include <iterator>
#include <algorithm>
#include <functional>
#include <utility>
#include <random>
#include <limits>
#include <unordered_map>

#include "DataFrame.h"

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

    Tree *parent, *left_subtree, *right_subtree; // temporary, used before I pack nodes in a vector
    size_t tree_size; // 1 + sizeof(left_subtree) + sizeof(right_subtree), valid only until tree is packed
    std::vector<Node> nodes; // vectorized tree - packed representation after I'm done with growing it

    // regression characteristics for prunning, valid before tree is packed
    double rss, sum, sum2; // RSS, sum, and sum of squares of dependent variable on training set
    // classification characteristics for prunning, valid before tree is packed
    std::unordered_map<long long, size_t> levelCounts; // cached counts for fast impurity calculations
    double gini, crossEntropy; // impurity of dependent variable on training set
    long long majorityVote; // dominant category of the set
    // below is an alias from one of the metrices above: gini/crossEntropy/rss
    double metric;
    // number of training entries - entries "seen" by this tree
    size_t set_size;

    // pack tree into a vector, free dynamically allocated memory, strip auxiliary characteristics
    size_t vectorize(std::vector<Node>& dest) {
        // sanity checks
        //  uninitialized?
        if( nodes.size() == 0 ) return 0;
        //  broken invariant (either vectorized or both subtrees present)?
        if( ( left_subtree == 0 && right_subtree != 0 ) ||
            ( left_subtree != 0 && right_subtree == 0 ) ) return 0;
        // already vectorized?
        if( tree_size == 0 ) return 0;

        // pre-order traversal
        dest.push_back(nodes[0]);
        Node& local_root = dest.back();

        size_t size = 1;

        // recure if not a terminal node
        if( left_subtree != 0 && right_subtree != 0 ){
            local_root.left_child  = dest.size();
            size += left_subtree->vectorize(dest);
            delete left_subtree;
            left_subtree = 0;

            local_root.right_child = dest.size();
            size += right_subtree->vectorize(dest);
            delete right_subtree;
            right_subtree = 0;
        }

        // no longer need this variable -> use it to indicate that this tree is already packed
        tree_size = 0;

        return size;
    }

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

    bool load(std::istream& input){
        return true; // to be implemented
    }
    bool save(std::ostream& output){
        // not packed? - error
        if( left_subtree != 0 || right_subtree != 0 )
            return false;
        for(unsigned int n=0; n<nodes.size(); n++)
            output << n
                   << "," << nodes[n].value
                   << "," << nodes[n].position
                   << "," << nodes[n].left_child
                   << "," << nodes[n].right_child
                   << std::endl;
        return true;
    }

    Tree(void) : parent(0), left_subtree(0), right_subtree(0), tree_size(0), nodes(0),
                 rss(0), sum(0), sum2(0),
                 levelCounts(), gini(0), crossEntropy(0),
                 majorityVote(0),
                 metric(0),
                 set_size(0) {}
    // tree is an owner of its subtrees
    ~Tree(void){
        if( left_subtree  ) delete left_subtree;
        if( right_subtree ) delete right_subtree;
    }
};

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
            double    bestMetricSoFar = std::numeric_limits<double>::max();;

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

    // weakest link pruning as prescribed in ESLII p.308
    void prune(Tree *tree, double alpha){
        std::vector<Tree*> candsForCollapse;
        double metricTotal = 0;

        // traverse the tree with local FIFO simulating stack of recursion
        std::queue<Tree*> fifo;
        fifo.push(tree);
        while( !fifo.empty() ){
            Tree *t = fifo.front();
            fifo.pop();
            Tree *t_l = t->left_subtree;
            Tree *t_r = t->right_subtree;
            if( t_l && t_r ){
                // look ahead for two leafs
                if( t_l->left_subtree == 0 && t_l->right_subtree == 0 &&
                    t_r->left_subtree == 0 && t_r->right_subtree == 0 ){
                    candsForCollapse.push_back(t);
                } else {
                    fifo.push(t_l);
                    fifo.push(t_r);
                }
            }
            if( t_l == 0 && t_r == 0 ){
                metricTotal += t->metric;
            }
        }

        // bookkeeping for number of collapses that'll lazily propagate up the tree
        std::unordered_map<Tree*,int> tree_size_decrease;

        std::function<bool(Tree*,Tree*)> greaterEq = [](Tree* i, Tree* j){ return i->metric >= j->metric; };

        std::make_heap(candsForCollapse.begin(), candsForCollapse.end(), greaterEq);

        while( metricTotal < alpha * tree->tree_size ){
            std::pop_heap(candsForCollapse.begin(), candsForCollapse.end(), greaterEq);
            Tree *t = candsForCollapse.back();
            candsForCollapse.pop_back();
            // metric (e.g. rss or gini) grows as I reduce model complexity
            metricTotal -= t->left_subtree->metric;
            metricTotal -= t->right_subtree->metric;
            metricTotal += t->metric;
            // new leaf has to become average/majorityVote rather than a split point
            switch( t->left_subtree->nodes[0].value.type ){
                case Variable::Continuous:
                    t->nodes[0].value.type = Variable::Continuous;
                    t->nodes[0].value.asFloating = t->sum/t->set_size;
                break;
                case Variable::Categorical:
                    t->nodes[0].value.type = Variable::Categorical;
                    t->nodes[0].value.asIntegral = t->majorityVote;
                break;
                default: break;
            }
            // collapsing t: chop-off the leafs
            delete t->left_subtree;
            t->left_subtree = 0;
            delete t->right_subtree;
            t->right_subtree = 0;
            tree_size_decrease[t] += 2;
            // parent may become a candidate for one of the next collapses
            Tree *p = t->parent;
            // already at the very top?
            if( p == 0 ) break;
            tree_size_decrease[p] += tree_size_decrease[t];
            if( p->tree_size - tree_size_decrease[p] == 3 ){
                candsForCollapse.push_back(p);
                std::push_heap(candsForCollapse.begin(), candsForCollapse.end(), greaterEq);
            }
        }

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
        rState.seed(0);
        const int nTrees = 1;
        for(unsigned int t=0; t<nTrees; t++){
            SplitVars vars( generateRandomSplitVars( df.getSchema(), predictorsIdx, std::floor(predictorsIdx.size()>15?predictorsIdx.size()/3:5) ) );//(unsigned int)sqrt(predictorsIdx.size()) ) );
//for(auto s : vars) cout << "s.first = "<<s.first << " s.second = "<< s.second << endl;
//            future<Tree> ft = async(std::launch::async, pickStrongestCuts, df, responseIdx, vars, sample(df.nrow(),df.nrow()*0.5));
            Tree *tree = findBestSplits(df, responseIdx, vars, sample(df.nrow(),df.nrow()*0.5));
            prune(tree,0.09);
            std::vector<Tree::Node> nodes;
            nodes.reserve(tree->tree_size);
            tree->vectorize(nodes);
            tree->nodes.swap(nodes);
tree->save(std::cout);
            ensemble.push_back( std::move(*tree) );
        }
    }

    RandomForest(void){}

//    load/save
};

#endif
