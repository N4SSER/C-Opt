//
// Created by nasser on 07/07/23.
//

#ifndef C_OPT_NSGA2_H
#define C_OPT_NSGA2_H

#include "GA.h"
#include <vector>
#include <algorithm>
#include <iostream>
struct Solution {
    int domination_count;
    std::vector<int> dominated_solutions;
    int rank;
};

class NSGA2 : public GA
{
public:
    Solution* nonDominatedSorting();
private:
    double** objectives{};
    int n_objectives{};

    bool dominates(const std::vector<double>& obj1, const std::vector<double>& obj2);
};


#endif //C_OPT_NSGA2_H
