#ifndef Tree_h
#define Tree_h

#include <map>
#include <queue>
#include <vector>
#include <algorithm>
#include <functional>
#include <utility>
#include <limits>
#include <memory>
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

    friend class TreeTrainer;

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

    // pack tree into a vector
    size_t vectorize(std::vector<Node>& dest) {
        // sanity checks
        //  uninitialized?
        if( nodes.size() == 0 ) return 0;
        //  broken invariant (either vectorized or both subtrees present)?
        if( ( left_subtree == 0 && right_subtree != 0 ) ||
            ( left_subtree != 0 && right_subtree == 0 ) ) return 0;

        // pre-order traversal
        dest.push_back(nodes[0]);
        // dest must have enough elements reserved otherwise one of the
        //  push_backs will invalidate local_root refs up the recursion chain
        Node& local_root = dest.back();

        size_t size = 1;

        // recure if not a terminal node
        if( left_subtree != 0 && right_subtree != 0 ){
            local_root.left_child  = dest.size();
            size += left_subtree->vectorize(dest);

            local_root.right_child = dest.size();
            size += right_subtree->vectorize(dest);
        }

        return size;
    }

    // function that eventually gets invoked by the end-used calling predict
    Variable traverseVectorized(const DataRow& row, const Node& root) const {
        // is it a leaf/terminal_node?
        if( root.left_child == 0 && root.right_child == 0 )
            return root.value;

        if( root.value.type == Variable::Continuous ){
            if( root.value.asFloating > row[root.position].asFloating )
                return traverseVectorized(row,nodes[root.left_child]);
            else
                return traverseVectorized(row,nodes[root.right_child]);
        }
        if( root.value.type == Variable::Categorical ){
            // only binary-level categories are managed
            if( root.value.asIntegral == row[root.position].asIntegral )
                return traverseVectorized(row,nodes[root.left_child]);
            else
                return traverseVectorized(row,nodes[root.right_child]);
        }
        // the root is neither Continuous nor Categorical -> error
        return Variable();
    }

    // analogue of the previous function that is handy while tree gets constructed (not vectorized)
    Variable traverse(const DataRow& row, const Tree* root) const {
        if( root == 0 ) return Variable(); // uninitialized Variable is an error sign

        // is it a leaf/terminal_node?
        if( root->left_subtree == 0 && root->right_subtree == 0 )
            return root->nodes[0].value;

        if( root->nodes[0].value.type == Variable::Continuous ){
            if( root->nodes[0].value.asFloating > row[root->nodes[0].position].asFloating )
                return traverse(row,root->left_subtree);
            else
                return traverse(row,root->right_subtree);
        }
        if( root->nodes[0].value.type == Variable::Categorical ){
            // only binary-level categories are managed
            if( root->nodes[0].value.asIntegral == row[root->nodes[0].position].asIntegral )
                return traverse(row,root->left_subtree);
            else
                return traverse(row,root->right_subtree);
        }
        // the root is neither Continuous nor Categorical -> error
        return Variable();
    }

    // weakest link pruning as prescribed in ESLII p.308
    //  I prune tree to a single node and return a series of intermediate/nested trees (pointers)
    //  each tree is a single node collapse smaller than the predecessor;
    //  trees are indexed by the alphas that would yeld these trees in cost-complexity
    //  optimization would I start with initial tree and the alphas
    std::map<double,std::shared_ptr<Tree>> prune(void){

        std::map<double,std::shared_ptr<Tree>> retval;

        // nothing to prune for a single-node tree return empty set
        if( tree_size == 1 ) return retval;

        std::vector<Tree*> candsForCollapse;
        double metricTotal = 0;

        // traverse the tree with local FIFO simulating stack of recursion
        std::queue<Tree*> fifo;
        fifo.push(this);
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
        // the idea is to take a note of tree size reduction, but not to spend time
        //  on updating every node yet as most of them won't be touched by pruning
        // (reduction only for the immediate parents of the pruned nodes is remembered)
        std::unordered_map<Tree*,int> tree_size_decrease;

        // "weakest link pruning: we successively collapse the internal node that
        //  produces the smallest per-node increase in ..."
        std::function<bool(Tree*,Tree*)> greaterEq = [](Tree* i, Tree* j){
            return i->metric - i->left_subtree->metric - i->right_subtree->metric >=
                   j->metric - j->left_subtree->metric - j->right_subtree->metric ;
        };

        // construct a priority queue out of the vector of candidates for pruning
        std::make_heap(candsForCollapse.begin(), candsForCollapse.end(), greaterEq);

        size_t totalSizeDecrease = 0;
        double alpha = 0;

        // for completenes store also the current unchanged tree (alpha = 0)
        std::shared_ptr<Tree> tree0(new Tree());
        tree0->nodes.reserve(tree_size);
        vectorize(tree0->nodes);
        retval[0] = tree0;

        // collapse nodes until there is nothing to collapse
        while( tree_size - totalSizeDecrease > 1 ){

            // first, estimate impact of pruning the weakest link on the cost-complexity 
            std::pop_heap(candsForCollapse.begin(), candsForCollapse.end(), greaterEq);
            Tree *t = candsForCollapse.back();
            candsForCollapse.pop_back();
            // metric (e.g. rss or gini) grows as I reduce model complexity
            metricTotal -= t->left_subtree->metric;
            metricTotal -= t->right_subtree->metric;
            metricTotal += t->metric;

            // by how much alpha should increase to balance the increase in metric
            alpha += (t->metric - t->left_subtree->metric - t->right_subtree->metric)/2.;

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
            tree_size_decrease.erase(t->left_subtree);
            t->left_subtree = 0;
            delete t->right_subtree;
            tree_size_decrease.erase(t->right_subtree);
            t->right_subtree = 0;
            tree_size_decrease[t] += 2;
            totalSizeDecrease += 2;
            // parent may become a candidate for one of the next collapses
            Tree *p = t->parent;
            // already at the very top?
            if( p == 0 ) break;
            tree_size_decrease[p] += tree_size_decrease[t];
            if( p->tree_size - tree_size_decrease[p] == 3 ){
                candsForCollapse.push_back(p);
                std::push_heap(candsForCollapse.begin(), candsForCollapse.end(), greaterEq);
            }

            std::shared_ptr<Tree> treeAlpha(new Tree());
            treeAlpha->nodes.reserve(tree_size-totalSizeDecrease);
            vectorize(treeAlpha->nodes);
            retval[alpha] = treeAlpha;
        }

        // pruned tree to just a single node?
        if( left_subtree == 0 && right_subtree == 0 ){
            tree_size = 1;
            return retval;
        }

        return retval;
    }

public:
    Variable predict(const DataRow& row) const {
        // is tree initialized? if not return default Variable as a sign of error
        if( nodes.size() == 0 ) return Variable(); 
        // is root node initialized?
        if( nodes[0].value.type == Variable::Unknown ) return Variable(); 
        // when tree is vectorized, I set tree_size to 0
        if( tree_size == 0 )
            return traverseVectorized(row,nodes[0]);
        // apparently, tree is not yet vectorized
        return traverse(row,this);
    }

    // return RSS or Gini metric
    double evaluateMetric(const DataFrame& df,
                          unsigned int responseIdx,
                          const std::vector<unsigned int>& subset)
    {
        double metric = std::numeric_limits<double>::max();
        // continuous response else categorical
        if( df.getLevels(responseIdx).size() == 0 ){
            double bias = 0, variance = 0;
            for(unsigned int row : subset){
                Variable p = predict( df[row] );
                double truth = df[row][responseIdx].asFloating;
                bias     +=  p.asFloating - truth;
                variance += (p.asFloating - truth) * (p.asFloating - truth);
            }
            metric = variance;
        } else {
            std::unordered_map<long long, std::pair<size_t,size_t>> matchMismatch;
            for(unsigned int row : subset){
                Variable p = predict( df[row] );
                long long trueLevel = df[row][responseIdx].asIntegral;
                if( p.asIntegral == trueLevel )
                    matchMismatch[ trueLevel ].first++;
                else
                    matchMismatch[ trueLevel ].second++;
            }
            double gini = 0;
            for(auto c : matchMismatch){
                size_t size = c.second.first + c.second.second;
                double p = double(c.second.first) / size;
                gini += p * (1 - p);
            }
            metric = gini;
        }
        return metric;
    }

    bool load(std::istream& input){
/*
0,(float)-1.01067,1,1,20
1,(float)-0.934883,0,2,9
2,(float)-1.04223,1,3,8
3,(float)-1.15702,1,4,7
4,(float)-1.09068,0,5,6
5,(int)2,0,0,0
*/
        return true; // to be implemented
    }
    bool save(std::ostream& output) const {
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

#endif
