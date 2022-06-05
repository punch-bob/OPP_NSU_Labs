#include <iostream>
#include "mpi.h"
#include <cmath>

void FillSetArrays(int N, int size, int* matrix_size, int* displs)
{
    for (int i = 0; i < size; ++i)
    {
        matrix_size[i] = N / size;
        displs[i] = i * (N / size);
    }
    for (int i = 0; i < N % size; ++i)
    {
        matrix_size[i]++;
        for (int j = i + 1; j < size; ++j)
        {
            displs[j]++;
        }
    }
    for (int i = 0; i < size; ++i)
    {
        matrix_size[i] *= N;
        displs[i] *= N;
    }
}

void SetVectors(int N, int size, int* matrix_size, int* displs, int* vector_size, int* v_displs)
{
    for (int i = 0; i < size; ++i)
    {
        vector_size[i] = matrix_size[i] / N;
        v_displs[i] = displs[i] / N;
    }
}

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
    double r = 0;
    double res = 0;
    for (int i = 0; i < N; ++i)
    {
        r += a[i] * b[i];
    }
    MPI_Allreduce(&r, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return res;
}

double ScalarMult(int N, double* a, double* b)
{
    double r = 0;
    double res = 0;
    for (int i = 0; i < N; ++i)
    {
        r += a[i] * b[i];
    }
    MPI_Allreduce(&r, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return res;
}

double CalcAlpha(int N, double* r, double* az, double* z)
{
    return (ScalarMult(N, r, r) / ScalarMult(N, az, z));
}

void MultMatrixVector(int N, int vector_size, double* A, double* v, double* res)
{
    double* tmp = (double*)calloc(N, sizeof(double));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < vector_size; ++j)
        {
            double* a = A + j * N;
            tmp[i] += a[i] * v[j];
        }
    }
    MPI_Reduce(tmp, res, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    free(tmp);
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
    return ((numerator * numerator) / b) < e;
}

void FoundSolution(int N, int* matrix_size, int* displs, double* A, double* b, double* x, double e, int size, int rank)
{
    auto vector_size = new int[size];
    auto v_displs = new int[size];
    SetVectors(N, size, matrix_size, displs, vector_size, v_displs);

    auto sub_b = new double[vector_size[rank]];
    MPI_Scatterv(b, vector_size, v_displs, MPI_DOUBLE, sub_b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto sub_x = new double[vector_size[rank]];
    MPI_Scatterv(x, vector_size, v_displs, MPI_DOUBLE, sub_x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto A_x = new double[N];
    auto sub_A_x = new double[vector_size[rank]];
    MultMatrixVector(N, vector_size[rank], A, sub_x, A_x);
    MPI_Scatterv(A_x, vector_size, v_displs, MPI_DOUBLE, sub_A_x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto sub_r = new double[vector_size[rank]];
    SubVectors(vector_size[rank], sub_b, sub_A_x, sub_r);

    auto sub_z = new double[vector_size[rank]];
    CopyVector(vector_size[rank], sub_r, sub_z);

    double b_lenght_squared = ScalarMult(vector_size[rank], sub_b, sub_b);
    b_lenght_squared *= b_lenght_squared;
    e *= e;
    while (!IsSolution(vector_size[rank], sub_r, b_lenght_squared, e))
    {
        auto A_z = new double[N];
        auto sub_A_z = new double[vector_size[rank]];
        MultMatrixVector(N, vector_size[rank], A, sub_z, A_z);
        MPI_Scatterv(A_z, vector_size, v_displs, MPI_DOUBLE, sub_A_z, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double alpha = CalcAlpha(vector_size[rank], sub_r, sub_A_z, sub_z);

        CalcNextX(vector_size[rank], sub_x, sub_z, alpha);

        auto new_r = new double[vector_size[rank]];
        CalcNextR(vector_size[rank], sub_r, sub_A_z, alpha, new_r);

        double betta = CalcBetta(vector_size[rank], new_r, sub_r);

        CalcNextZ(vector_size[rank], sub_z, new_r, betta);

        CopyVector(vector_size[rank], new_r, sub_r);
        delete[] A_z;
        delete[] sub_A_z;
        delete[] new_r;
    }
    delete[] A_x;
    delete[] sub_A_x;
    delete[] sub_r;
    delete[] sub_b;
    delete[] sub_z;
    MPI_Allgatherv(sub_x, vector_size[rank], MPI_DOUBLE, x, vector_size, v_displs, MPI_DOUBLE, MPI_COMM_WORLD);
    delete[] vector_size;
    delete[] v_displs;
    delete[] sub_x;
}

int main(int argc, char** argv)
{
    int N, size, rank;
    double e = 0.00001;
    MPI_Init(&argc, &argv);
    double starttime, endtime;
    starttime = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    N = atoi(argv[1]);
    auto matrix_size = new int[size];
    auto displs = new int[size];
    FillSetArrays(N, size, matrix_size, displs);
    auto A = new double[N * N];
    auto sub_A = new double[matrix_size[rank]];
    auto b = new double[N];
    auto x = new double[N];
    auto sub_answ = new double[matrix_size[rank] / N];
    if (rank == 0)
    {
        FillMatrix(N, A);
        FillVector(N, b);
        FillVector(N, x);
    }
    MPI_Scatterv(A, matrix_size, displs, MPI_DOUBLE, sub_A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    FoundSolution(N, matrix_size, displs, sub_A, b, x, e, size, rank);
    endtime = MPI_Wtime();
    if (rank == 0)
    {
        printf("Time: %lf", endtime - starttime);
    }
    auto vector_size = new int[size];
    auto v_displs = new int[size];
    auto tmp = new double[N];
    SetVectors(N, size, matrix_size, displs, vector_size, v_displs);
    MPI_Scatterv(x, vector_size, v_displs, MPI_DOUBLE, sub_answ, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MultMatrixVector(N, vector_size[rank], sub_A, sub_answ, tmp);
    if (rank == 0)
    {
        int counter = 0;
        std::cout << "\n======================\n";
        for (int i = 0; i < N; ++i)
        {
            if ((int)(b[i] - tmp[i]) != 0) counter++;
        }
        std::cout << counter << std::endl;
    }
    delete[] x;
    delete[] b;
    delete[] A;
    delete[] matrix_size;
    delete[] displs;
    delete[] sub_A;
    delete[] sub_answ;
    delete[] vector_size;
    delete[] v_displs;
    delete[] tmp;
    MPI_Finalize();
    return 0;
}