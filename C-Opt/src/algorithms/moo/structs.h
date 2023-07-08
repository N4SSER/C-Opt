//
// Created by nasser on 07/07/23.
//

#ifndef C_OPT_STRUCTS_H
#define C_OPT_STRUCTS_H

struct Solution
{
    int domination_count;
    std::vector<int> dominated_solutions;
    int rank;
};

struct Front
{
    std::vector<int> dominated_solutions;
    std::vector<int> crowding_distances;

};

#endif //C_OPT_STRUCTS_H
