//
// Created by nasser on 06/07/23.
//

#ifndef C_OPT_GA_H
#define C_OPT_GA_H

#include <iostream>
#include <string>
#include <cmath>
#include <cstring>
#include <random>

class GA
{

protected:
    double crossover_probability;
    double mutation_rate;
    int population_size;
    int offspring_size;
    int n_dim;
    double** population;
    double** offsprings;
    double* upper_bound;
    double* lower_bound;

    virtual void crossover();
    virtual void mutate(std::string** population, int size);

private:
    std::string crossover(const std::string& parent_1, const std::string& parent_2);
    std::string* _encode(const double* solution);
    std::string** encode(double** population, int size);
    double* decode(const std::string* encoded_solution);
    double **decode(std::string** encoded_population, int size);
    std::string toBin(int encoded_value);
    int toInt(const std::string& bin_value);
};


#endif //C_OPT_GA_H
