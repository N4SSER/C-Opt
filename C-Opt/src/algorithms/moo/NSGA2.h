//
// Created by nasser on 07/07/23.
//

#ifndef C_OPT_NSGA2_H
#define C_OPT_NSGA2_H

#include "../GA.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include "structs.h"
class NSGA2 : public GA
{
public:
    Solution* nonDominatedSorting();
    std::vector<std::vector<double>> crowdingDistances(std::vector<std::vector<int>> fronts);
    std::vector<std::vector<int>> getFronts(const Solution *solutions);
private:
    double** objectives;
    int n_objectives;
    double** findMinMax(double** objectives, int population_size);
    double** subPop(int size, const std::vector<int>& indexes);
    bool dominates(const std::vector<double>& obj1, const std::vector<double>& obj2);

};


#endif //C_OPT_NSGA2_H
