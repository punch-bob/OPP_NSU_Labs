#include <iostream>
#include <cmath>
#include <omp.h>

void FillMatrix(int N, double* A)
{
    #pragma omp for
    for (int i = 0; i < N; ++i)
    {
        double* a = A + i * N;
        for (int j = i; j < N; ++j)
        {
            double rand = double(std::rand() % 10000) / 100;
            int sign = std::rand() % 2;
            if (sign == 1)
            {
                rand *= -1;
            }
            a[j] = rand;
            A[j * N + i] = rand;
        }
        a[i] += 200;
    }
}

void FillVector(int N, double* v)
{
    #pragma omp for
    for (int i = 0; i < N; ++i)
    {
        double rand = double(std::rand() % 10000) / 100;
        int sign = std::rand() % 2;
        if (sign == 1)
        {
            rand *= -1;
        }
        v[i] = rand;
    }
}

void SubVectors(int N, double* a, double* b, double* c)
{
    #pragma omp for
    for (int i = 0; i < N; ++i)
    {
        c[i] = a[i] - b[i];
    }
}

void CopyVector(int N, double* a, double* b)
{
    #pragma omp for
    for (int i = 0; i < N; ++i)
    {
        b[i] = a[i];
    }
}

double ScalarMult(int N, double* a, double* b)
{
    double res = 0;
    #pragma omp parallel for reduction(+: res)
    for (int i = 0; i < N; ++i)
    {
        res += a[i] * b[i];
    }
    return res;
}

double CalcAlpha(int N, double module_r, double* az, double* z)
{
    return (module_r / ScalarMult(N, az, z));
}

void MultMatrixVector(int N, double* A, double* v, double* res)
{
    #pragma omp for
    for (int i = 0; i < N; ++i)
    {
        double* a = A + i * N;
        res[i] = 0;
        for (int j = 0; j < N; ++j)
        {
            res[i] += a[j] * v[j];
        }
    }
}

void MultVectorScalar(int N, double c, double* a, double* b)
{
    #pragma omp for
    for (int i = 0; i < N; ++i)
    {
        b[i] = a[i] * c;
    }
}

void CalcNextX(int N, double* x, double* z, double alpha)
{
    #pragma omp for
    for (int i = 0; i < N; ++i)
    {
        x[i] = x[i] + z[i] * alpha;
    }
}

void CalcNextR(int N, double* r, double* A_z, double alpha, double* new_r)
{
    #pragma omp for
    for (int i = 0; i < N; ++i)
    {
        new_r[i] = r[i] - A_z[i] * alpha;
    }
}

void CalcNextZ(int N, double* z, double* r, double betta)
{
    #pragma omp for
    for (int i = 0; i < N; ++i)
    {
        z[i] = r[i] + betta * z[i];
    }
}

bool IsSolution(double module_r, double b, double e)
{
    return (((module_r * module_r) / b) < e);
}

void FoundSolution(int N, double* A, double* b, double* x, double e)
{
    auto r = new double[N];
    auto z = new double[N];
    auto A_x = new double[N];
    auto A_z = new double[N];
    auto new_r = new double[N];
    double alpha;
    double betta;
    double module_new_r = 0;
    double module_b;
    double module_r;
    double alpha_z;
#pragma omp parallel
    {
        MultMatrixVector(N, A, x, A_x);
        SubVectors(N, b, A_x, r);
        CopyVector(N, r, z);
        module_b = ScalarMult(N, b, b);
        module_r = ScalarMult(N, r, r);
    }
    module_b *= module_b;
    e *= e;
#pragma omp parallel
    {
        while (!IsSolution(module_r, module_b, e))
        {
            MultMatrixVector(N, A, z, A_z);
            alpha = CalcAlpha(N, module_r, A_z, z);
            CalcNextX(N, x, z, alpha);
            CalcNextR(N, r, A_z, alpha, new_r);
            module_new_r = ScalarMult(N, new_r, new_r);
            betta = module_new_r / module_r;
            CalcNextZ(N, z, new_r, betta);
            CopyVector(N, new_r, r);
            module_r = module_new_r;
        }
    }
    delete[] new_r;
    delete[] A_z;
    delete[] A_x;
    delete[] z;
    delete[] r;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Not enought arguments!" << std::endl;
        return 0;
    }
    int N = atoi(argv[1]);
    double e = 0.00000001;
    auto A = new double[N * N];
    auto b = new double[N];
    auto x = new double[N];
    double start, end;
#pragma omp parallel
    {
        FillMatrix(N, A);
        FillVector(N, b);
        FillVector(N, x);
    }
    start = omp_get_wtime();
    FoundSolution(N, A, b, x, e);
    end = omp_get_wtime();
    auto res = new double[N];
    MultMatrixVector(N, A, x, res);
    double misstakes = 0;
    for (int i = 0; i < N; ++i)
    {
        if (int(b[i] - res[i]) != 0) misstakes++;
    }
    std::cout << "============+" << std::endl;
    std::cout << "Misstakes: " << misstakes << std::endl;
    std::cout << "============+" << std::endl;
    std::cout << "Times: " << end - start << std::endl;
    std::cout << "============+" << std::endl;
    delete[] res;
    delete[] x;
    delete[] b;
    delete[] A;
    return 0;
}
