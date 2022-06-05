#include <iostream>
#include "mpi.h"
#include <list>
#include <cstring>
#include <iterator>

void setArray(int sizeX, bool* A)
{
    A[0 * sizeX + 1] = true;
    A[1 * sizeX + 2] = true;
    A[2 * sizeX + 0] = true;
    A[2 * sizeX + 1] = true;
    A[2 * sizeX + 2] = true;
}

void setArrays(int* displs, int* sendcounts, int sizeY, int sizeX, int size)
{
    int partSize = (int)(sizeY / size);
    for (int i = 0; i < size; ++i)
    {       
        sendcounts[i] = partSize + 2;
    }
    sendcounts[0] -= 1;
    sendcounts[size - 1] -= 1;

    for (int i = 0; i < sizeY % size; ++i)
    {
        sendcounts[i]++;
    }

    int totalDispls = 0;
    for (int i = 0; i < size; ++i)
    {
        displs[i] = totalDispls * sizeX;
        totalDispls += sendcounts[i] - 2;
        sendcounts[i] *= sizeX;
    }
}

void printMatr(bool* A, int Y, int X)
{
    for (int i = 0; i < Y; ++i)
    {
        bool* a = A + i * X;
        for (int j = 0; j < X; ++j)
        {
            std::cout << a[j];
        }
        std::cout << std::endl;
    }
    std::cout << "\n\n";
}


void setNeighbors(int size, int rank, int* upperNeighbor, int* lowerNeighbor)
{

    *upperNeighbor = rank - 1;
    *lowerNeighbor = rank + 1;

    if (rank == 0)
    {
        *upperNeighbor = size - 1;
    }
    if (rank == size - 1)
    {
        *lowerNeighbor = 0;
    }
}

bool compareMatrix(bool* A, bool* B, int matrSize)
{
    for (int i = 0; i < matrSize; ++i)
    {
        if (A[i] != B[i])
        {
            return false;
        }
    }
    return true;
}

bool* calcStopVector(std::list<bool*> prevStates, bool* partA, int matrSize, int vectorSize)
{
    auto stopVector = new bool[vectorSize];
    auto it = prevStates.begin();
    for (int i = 0; i < vectorSize; ++i)
    {
        stopVector[i] = compareMatrix(*it, partA, matrSize);
        it++;
    }
    return stopVector;
}

bool isEnd(int sizeY, int sizeX, bool* stopMatr)
{
    for (int i = 0; i < sizeX; ++i)
    {
        int count = 0;
        for (int j = 0; j < sizeY; ++j)
        {
            if (stopMatr[j * sizeX + i]) count++;
        }
        if (count == sizeY)
        {
            return true;
        }
    }
    return false;
}

bool updateState(bool prev, int cnt)
{
    if (prev)
    {
        if (cnt < 2 || cnt > 3) return false;
    }
    else
    {
        if (cnt == 3) return true;
    }
    return prev;
}

int get(bool* partA, int sizeX, int coordY, int coordX) 
{
    return (int)partA[coordY * sizeX + coordX % sizeX];
}

void calcNext(int sizeY, int sizeX, bool* partA, bool* nextPartA)
{
    for (int i = 1; i < sizeY - 1; ++i)
    {
        for (int j = 0; j < sizeX; j++)
        {
            int aliveNear = get(partA, sizeX, i, (j + 1)) + get(partA, sizeX, i, (j + sizeX - 1)) + get(partA, sizeX, i + 1, (j + 1)) + get(partA, sizeX, i + 1, (j + sizeX - 1)) +
                            get(partA, sizeX, i - 1, (j + 1)) + get(partA, sizeX, i - 1, (j + sizeX - 1)) + get(partA, sizeX, i + 1, j) + get(partA, sizeX, i - 1, j);

            nextPartA[i * sizeX + j] = updateState(partA[i * sizeX + j], aliveNear);
        }
    }
}

void lifeGame(int size, int rank, bool* A, int sizeY, int sizeX)
{
    auto displs = new int[size];
    auto sendcounts = new int[size];
    setArrays(displs, sendcounts, sizeY, sizeX, size);

    bool* partA;
    if (rank == 0 || rank == size - 1)
    {
        partA = new bool[sendcounts[rank] + sizeX];
    }
    else
    {
        partA = new bool[sendcounts[rank]];
    }
    MPI_Scatterv(A, sendcounts, displs, MPI_C_BOOL, partA, sendcounts[rank], MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::copy(partA, &partA[sendcounts[rank]], &partA[sizeX]);
        memset(partA, false, sizeX);
    }

    if (rank == size - 1)
    {
        memset(&partA[sendcounts[rank]], false, sizeX);
    }

    if (rank == 0 || rank == size - 1)
    {
        sendcounts[rank] += sizeX;
    }

    int upperNeighbor, lowerNeighbor;
    setNeighbors(size, rank, &upperNeighbor, &lowerNeighbor);

    std::list<bool*> prevStates;
    auto nextPartA = new bool[sendcounts[rank]];

    auto copyPartA = new bool[sendcounts[rank]];
    std::copy(partA, &partA[sendcounts[rank]], copyPartA);

    int iterID = 0;
    int partSizeY = sendcounts[rank] / sizeX;
    bool end = false;
    while (!end)
    {
        memset(nextPartA, false, sendcounts[rank]);
        copyPartA = new bool[sendcounts[rank]];
        std::copy(partA, &partA[sendcounts[rank]], copyPartA);
        prevStates.push_back(copyPartA);
        iterID++;
        MPI_Request upperSendRequest;
        MPI_Request lowerSendRequest;
        MPI_Request upperRecvRequest;
        MPI_Request lowerRecvRequest;
        MPI_Isend(&partA[sizeX], sizeX, MPI_C_BOOL, upperNeighbor, 0, MPI_COMM_WORLD, &upperSendRequest);
        MPI_Isend(&partA[(partSizeY - 2) * sizeX], sizeX, MPI_C_BOOL, lowerNeighbor, 0, MPI_COMM_WORLD, &lowerSendRequest);
        MPI_Irecv(partA, sizeX, MPI_C_BOOL, upperNeighbor, 0, MPI_COMM_WORLD, &upperRecvRequest);
        MPI_Irecv(&partA[(partSizeY - 1) * sizeX], sizeX, MPI_C_BOOL, lowerNeighbor, 0, MPI_COMM_WORLD, &lowerRecvRequest);

        bool* stopVector;
        bool* stopMatr;
        MPI_Request stopVectorRequest;
        if (iterID > 1)
        {
            stopMatr = new bool[(iterID - 1) * size];
            stopVector = calcStopVector(prevStates, partA, sendcounts[rank], iterID - 1);
            MPI_Iallgather(stopVector, iterID - 1, MPI_C_BOOL, stopMatr, iterID - 1, MPI_C_BOOL, MPI_COMM_WORLD, &stopVectorRequest);
        }

        calcNext(partSizeY - 2, sizeX, &partA[sizeX], &nextPartA[sizeX]);
        MPI_Wait(&upperSendRequest, MPI_STATUS_IGNORE);
        MPI_Wait(&upperRecvRequest, MPI_STATUS_IGNORE);
        calcNext(3, sizeX, partA, nextPartA);
        MPI_Wait(&lowerSendRequest, MPI_STATUS_IGNORE);
        MPI_Wait(&lowerRecvRequest, MPI_STATUS_IGNORE);
        calcNext(3, sizeX, &partA[(partSizeY - 3) * sizeX], &nextPartA[(partSizeY - 3) * sizeX]);
        std::copy(nextPartA, &nextPartA[sendcounts[rank]], partA);

        if (iterID > 1)
        {
            MPI_Wait(&stopVectorRequest, MPI_STATUS_IGNORE);
            end = isEnd(size, iterID - 1, stopMatr);
            delete[] stopMatr;
            delete[] stopVector;
        }
    }

    delete[] displs;
    delete[] sendcounts;
    delete[] nextPartA;
    delete[] partA;
}

int main(int argc, char* argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 3)
    {
        std::cerr << "Not enough args!" << std::endl;
    }
    int sizeY = std::atoi(argv[1]);
    int sizeX = std::atoi(argv[2]);
    bool* A = new bool[sizeY * sizeX];
    if (rank == 0)
    {
        memset(A, false, sizeX * sizeY);
        setArray(sizeX, A);
    }

    double start, end;
    start = MPI_Wtime();
    lifeGame(size, rank, A, sizeY, sizeX);
    end = MPI_Wtime();
    if (rank == 0)
    {
        std::cout << "Times taken: " << end - start << "sec.\n";
        delete[] A;
    }

    MPI_Finalize();
}