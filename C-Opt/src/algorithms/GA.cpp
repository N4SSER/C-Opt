//
// Created by nasser on 06/07/23.
//

#include "GA.h"

GA::GA(int population_size, int offspring_size, int n_dim, double *upper_bound, double *lower_bound,
       double crossover_probability, double mutation_rate)
{
    this->population_size = population_size;
    this->offspring_size = offspring_size;
    this->n_dim = n_dim;
    this->upper_bound = upper_bound;
    this->lower_bound = lower_bound;
    this->crossover_probability = crossover_probability;
    this->mutation_rate = mutation_rate;

    init();
}
void GA::crossover()
{
    std::string** population = encode(this->population,population_size);
    std::string** offsprings;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    offsprings = new std::string*[offspring_size];

    for (int i = 0; i < offspring_size; i++)
        offsprings[i] = new std::string[n_dim];

    for (int i = 0; i < offspring_size; i += 2) {
        std::string* parent1 = population[i % population_size];
        std::string* parent2 = population[(i + 1) % population_size];

        if (dis(gen) > crossover_probability)
        {
            offsprings[i] = parent1;
            offsprings[i + 1] = parent2;
        }
        else
        {
            offsprings[i] = new std::string[offspring_size];
            offsprings[i + 1] = new std::string[offspring_size];
            for (int j = 0; j < offspring_size; j++) {
                offsprings[i][j] = crossover(parent1[j], parent2[j]);
                offsprings[i + 1][j] = crossover(parent2[j], parent1[j]);
            }
        }
    }
    std::memcpy(this->offsprings, decode(offsprings,offspring_size),sizeof(double*)*offspring_size);
}

void GA::mutate(std::string** population, int size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < n_dim; j++)
        {
            if (dis(gen) <= mutation_rate)
            {
                for (char& bit : population[i][j])
                    bit = (bit == '0') ? '1' : '0';
            }
        }
    }
}


int GA::toInt(const std::string& bin_value)
{
    int result = 0;
    int power = 1;

    for (int i = bin_value.length() - 1; i >= 0; --i)
    {
        if (bin_value[i] == '1')
            result += power;
        power *= 2;
    }

    return result;
}

std::string GA::toBin(int encoded_value)
{
    if (encoded_value == 0)
        return "0";

    std::string binary_string;
    while (encoded_value > 0)
    {
        binary_string = (encoded_value % 2 == 0 ? "0" : "1") + binary_string;
        encoded_value /= 2;
    }

    return binary_string;
}

std::string** GA::encode(double **population, int size)
{
    std::string** encoded_population = new std::string*[size];

    for (int i = 0; i < size; ++i)
        encoded_population[i] = new std::string[n_dim];

    for(int i =0; i< size;i++)
    {
        memcpy(encoded_population[i], _encode(population[i]), n_dim * sizeof(std::string));
    }

    return encoded_population;
}

double** GA::decode(std::string **encoded_population, int size)
{
    double** decoded_population = new double*[size];

    for (int i = 0; i < size; ++i)
        decoded_population[i] = new double[n_dim];

    for(int i =0; i< size;i++)
    {
        memcpy(decoded_population[i], decode(encoded_population[i]), n_dim * sizeof(std::string));
    }
}

std::string *GA::_encode(const double *solution)
{
    std::string* encoded_solution = (std::string*)malloc(n_dim*(sizeof (std::string*)));
    for(int i =0; i<n_dim; i++)
    {
        double range = upper_bound[i] - lower_bound[i];
        int m = ceil(log2(range * 100));
        int encoded_value = floor((solution[i] - lower_bound[i]) / range * pow(2, m))-1;
        encoded_solution[i] = toBin(encoded_value);
    }

    return encoded_solution;
}

double *GA::decode(const std::string* encoded_solution)
{
    double* decoded_solution = (double*) malloc(n_dim*(sizeof (double*)));
    for(int i =0; i<n_dim; i++)
    {
        double range = upper_bound[i] - lower_bound[i];
        int m = ceil(log2(range * 100));
        double decoded_value = (static_cast<double>(toInt(encoded_solution[i])) / pow(2, m)) * range + lower_bound[i];
        decoded_solution[i] = decoded_value;
    }

    return decoded_solution;
}

std::string GA::crossover(const std::string &parent_1, const std::string &parent_2)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> indexDist(1, parent_1.length() - 1);

    int crossoverPoint = indexDist(gen);

    std::string child = parent_1.substr(0, crossoverPoint) + parent_2.substr(crossoverPoint);
    return child;
}

void GA::init()
{
    population = new double*[population_size];
    for (int i = 0; i < population_size; i++)
        population[i] = new double[n_dim];

    offsprings = new double*[offspring_size];
    for (int i = 0; i < offspring_size; i++)
        offsprings[i] = new double[n_dim];

    for(int i =0; i < population_size; i++)
    {
        for(int j = 0; i<n_dim;j++)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dist(lower_bound[j], upper_bound[j]);
            population[i][j] = dist(gen);
        }
    }
}

