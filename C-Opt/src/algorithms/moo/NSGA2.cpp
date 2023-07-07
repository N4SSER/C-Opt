//
// Created by nasser on 07/07/23.
//

#include "NSGA2.h"

Solution* NSGA2::nonDominatedSorting()
{
    auto* solutions = (Solution*)malloc(population_size*sizeof(Solution*));

    for (int i = 0; i < population_size; ++i) {
        solutions[i].domination_count = 0;
        solutions[i].rank = 0;
        for (int j = 0; j < population_size; ++j)
        {
            if (dominates(std::vector<double>(objectives[j], objectives[j] + n_objectives),
                          std::vector<double>(objectives[i], objectives[i] + n_objectives)))
            {
                solutions[i].dominated_solutions.push_back(j);
            } else if (dominates(std::vector<double>(objectives[i], objectives[i] + n_objectives),
                                 std::vector<double>(objectives[j], objectives[j] + n_objectives)))
            {
                solutions[i].domination_count++;
            }
        }
        if (solutions[i].domination_count == 0)
            solutions[i].rank = 1;
    }

    int rank = 1;
    std::vector<int> currentFront;
    while (!currentFront.empty() || rank == 1)
    {
        currentFront.clear();
        for (int i = 0; i < population_size; ++i)
        {
            if (solutions[i].rank == rank)
                currentFront.push_back(i);
        }

        for (int i : currentFront)
        {
            for (int j : solutions[i].dominated_solutions)
            {
                if (--solutions[j].domination_count == 0)
                    solutions[j].rank = rank + 1;

            }
        }
        rank++;
    }

    return solutions;
}

bool NSGA2::dominates(const std::vector<double> &obj1, const std::vector<double> &obj2) {
    bool strictly_better = false;
    for (size_t i = 0; i < obj1.size(); ++i) {
        if (obj1[i] > obj2[i])
            return false;
        else if (obj1[i] < obj2[i])
            strictly_better = true;
    }
    return strictly_better;
}
