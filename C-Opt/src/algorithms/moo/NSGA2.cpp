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

bool NSGA2::dominates(const std::vector<double> &obj1, const std::vector<double> &obj2)
{
    bool strictly_better = false;
    for (size_t i = 0; i < obj1.size(); ++i) {
        if (obj1[i] > obj2[i])
            return false;
        else if (obj1[i] < obj2[i])
            strictly_better = true;
    }
    return strictly_better;
}

std::vector<std::vector<int>> NSGA2:: getFronts(const Solution* solutions)
{
    int maxRank = 0;
    for (int i = 0; i < population_size; ++i)
        maxRank = std::max(maxRank, solutions[i].rank);

    std::vector<std::vector<int>> fronts(maxRank);
    for (int i = 0; i < population_size; ++i)
        fronts[solutions[i].rank - 1].push_back(i);

    return fronts;
}

std::vector<std::vector<double>> NSGA2::crowdingDistances(std::vector<std::vector<int>> fronts) {
    std::vector<std::vector<double>> crowding_distances(fronts.size());
    for(auto& front : fronts)
    {
        double** min_max = findMinMax(population,population_size);
        for(int k = 1; k< front.size()-1;k++)
        {
            auto CD = 0.0;
            for(int m = 0; m< n_objectives;m++)
            {
                 auto b = objectives[front[k+1]][m] - objectives[front[k-1]][m];
                 auto a = min_max[m][1] - min_max[m][0];
                 CD += b/a;
            }
            crowding_distances[k].push_back(CD);
        }
    }
    return crowding_distances;
}

double **NSGA2::findMinMax(double** objectives, int population_size)
{
    double** result = new double*[n_objectives];

    for (int i = 0; i < n_objectives; i++)
    {
        double maxVal = objectives[0][i];
        double minVal = objectives[0][i];

        for (int j = 1; j < population_size; j++)
        {
            if (objectives[j][i] > maxVal)
                maxVal = objectives[j][i];
            if (objectives[j][i] < minVal)
                minVal = objectives[j][i];
        }
        result[i] = new double[2];
        result[i][1] = maxVal;
        result[i][0] = minVal;
    }

    return result;
}

double **NSGA2::subPop(int size, const std::vector<int> &indexes)
{
    int subRows = indexes.size();
    double** subMatrix = new double*[subRows];

    for (int i = 0; i < subRows; i++) {
        subMatrix[i] = new double[n_objectives];

        int rowIndex = indexes[i];
        for (int j = 0; j < n_objectives; j++) {
            subMatrix[i][j] = objectives[rowIndex][j];
        }
    }

    return subMatrix;
}
