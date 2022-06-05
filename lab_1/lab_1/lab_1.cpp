#include <iostream>
#include <cmath>
#include <sys/times.h>
#include <unistd.h>

void FillMatrix(int N, double* A)
{
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
    for (int i = 0; i < N; ++i)
    {
        c[i] = a[i] - b[i];
    }
}

void CopyVector(int N, double* a, double* b)
{
    for (int i = 0; i < N; ++i)
    {
        b[i] = a[i];
    }
}

double ScalarMult(int N, double* a, double* b)
{
    double res = 0;
    for (int i = 0; i < N; ++i)
    {
        res += a[i] * b[i];
    }
    return res;
}

double CalcAlpha(int N, double* r, double* az, double* z)
{
    return (ScalarMult(N, r, r) / ScalarMult(N, az, z));
}

void MultMatrixVector(int N, double *A, double *v, double *res)
{
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
    for (int i = 0; i < N; ++i)
    {
        b[i] = a[i] * c;
    }
}

void CalcNextX(int N, double* x, double* z, double alpha)
{
    auto alpha_z = new double[N];
    MultVectorScalar(N, alpha, z, alpha_z);
    for (int i = 0; i < N; ++i)
    {
        x[i] = x[i] + alpha_z[i];
    }
    delete[] alpha_z;
}

void CalcNextR(int N, double* r, double* A_z, double alpha, double* new_r)
{
    MultVectorScalar(N, alpha, A_z, A_z);
    SubVectors(N, r, A_z, new_r);
}

void CalcNextZ(int N, double* z, double* r, double betta)
{
    for (int i = 0; i < N; ++i)
    {
        z[i] = r[i] + betta * z[i];
    }
}

double CalcBetta(int N, double* new_r, double* prev_r)
{
    return ScalarMult(N, new_r, new_r) / ScalarMult(N, prev_r, prev_r);
}

bool IsSolution(int N, double* r, double b, double e)
{
    double numerator = ScalarMult(N, r, r);
    return (((numerator * numerator) / b) < e);
}

void FoundSolution(int N, double* A, double* b, double* x, double e)
{
    auto r = new double[N];
    auto z = new double[N];
    auto A_x = new double[N];
    MultMatrixVector(N, A, x, A_x);
    SubVectors(N, b, A_x, r);
    CopyVector(N, r, z);
    int counter = 0;
    double module_b = ScalarMult(N, b, b);
    module_b *= module_b;
    e *= e;
    while (!IsSolution(N, r, module_b, e))
    {
        auto A_z = new double[N];
        MultMatrixVector(N, A, z, A_z);
        double alpha = CalcAlpha(N, r, A_z, z);
        CalcNextX(N, x, z, alpha);
        auto new_r = new double[N];
        CalcNextR(N, r, A_z, alpha, new_r);
        double betta = CalcBetta(N, new_r, r);
        CalcNextZ(N, z, new_r, betta);
        CopyVector(N, new_r, r);
        delete[] new_r;
        delete[] A_z;
        counter++;
    }
    delete[] A_x;
    delete[] z;
    delete[] r;
    std::cout << "Total iters: " << counter << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        perror("not enought arguments!\n");
        return 0;
    }
    struct tms start, end;
    times(&start);
    int N = atoi(argv[1]);
    double e = 0.00001;
    auto A = new double[N * N];
    auto b = new double[N];
    auto x = new double[N];
    FillMatrix(N, A);
    FillVector(N, b);
    FillVector(N, x);
    FoundSolution(N, A, b, x, e);
    times(&end);
    double clocks = end.tms_utime - start.tms_utime;
    printf("Times: %lf", clocks / sysconf(_SC_CLK_TCK));
    delete[] x;
    delete[] b;
    delete[] A;
    return 0;
}