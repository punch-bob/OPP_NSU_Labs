#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <cmath>
#include "mpi.h"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <vector>

#define NO_TASK -1

int tasksOnIter = 40000;
int totalIters = 10;
int L = 1000;

int size;
int rank;

std::vector<int> taskList;
int tasksComplete = 0;
double globalRes = 0;
double summImbalance = 0;
bool workerFinish = false;
bool* rankStatus;
bool iAmFree = false;
pthread_mutex_t mutex;
pthread_t workerThread;
pthread_t receiverThread;

void fillTaskLists(int iterCounter)
{
    for (int i = 0; i < tasksOnIter; ++i)
    {
        taskList.push_back(abs(50 - i % 100) * abs(rank - (iterCounter % size)) * L);
    }
}

void work()
{
    while (true)
    {
        pthread_mutex_lock(&mutex);
        if (taskList.empty())
        {
            pthread_mutex_unlock(&mutex);
            break;
        }
        int repeatNum = taskList.back();
        taskList.pop_back();
        pthread_mutex_unlock(&mutex);
        for (int j = 0; j < repeatNum; ++j)
        {
            globalRes += sin(j);
        }
        tasksComplete++;
    }
    pthread_mutex_lock(&mutex);
    iAmFree = true;
    pthread_mutex_unlock(&mutex);
}

void* worker(void* attr)
{
    double start, end, iterTime, lIter, sIter;
    for (int i = 0; i < totalIters; ++i)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        pthread_mutex_lock(&mutex);
        iAmFree = false;
        workerFinish = false;
        pthread_mutex_unlock(&mutex);
        fillTaskLists(i);
        tasksComplete = 0;
        start = MPI_Wtime();
        work();
        while (true)
        {
            pthread_mutex_lock(&mutex);
            bool workerFinishTmp = workerFinish;
            pthread_mutex_unlock(&mutex);
            if (workerFinishTmp) break;

            work();
        }
        end = MPI_Wtime();

        pthread_mutex_lock(&mutex);
        iAmFree = false;
        pthread_mutex_unlock(&mutex);

        iterTime = end - start;
        MPI_Allreduce(&iterTime, &lIter, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iterTime, &sIter, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        for (int j = 0; j < size; ++j)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == j)
            {
                std::cout << "+==================================+" << std::endl;
                std::cout << " Process number: " << rank << " on iteration: " << i << std::endl;
                std::cout << "+==================================+" << std::endl;
                std::cout << " Total tasks completed: " << tasksComplete << std::endl;
                std::cout << "+==================================+" << std::endl;
                std::cout << " Work result: " << globalRes << std::endl;
                std::cout << "+==================================+" << std::endl;
                std::cout << " Iteration time: " << std::fixed << std::setprecision(3) << iterTime << " sec." << std::endl;
                std::cout << "+==================================+" << std::endl << std::endl;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
        {
            std::cout << std::endl << i << " iteration statistics:" << std::endl;
            std::cout << "+==================================+" << std::endl;
            std::cout << " Time of imbalance: " << lIter - sIter << " sec." << std::endl;
            std::cout << "+==================================+" << std::endl;
            std::cout << " The proportion of imbalance: " << std::fixed << std::setprecision(2) << (lIter - sIter) / lIter * 100 << "%" << std::endl;
            std::cout << "+==================================+" << std::endl << std::endl << std::endl << std::endl;
            summImbalance += (lIter - sIter) / lIter;
        }
    }
    pthread_exit(nullptr);
}

void* receiver(void* attr)
{
    bool status;
    int iterationsFinished = 0;
    int newTasksList[size];
    while (iterationsFinished < totalIters)
    {
        MPI_Allgather(&iAmFree, 1, MPI_C_BOOL, rankStatus, 1, MPI_C_BOOL, MPI_COMM_WORLD);

        for (int i = 0; i < size; ++i)
        {
            pthread_mutex_lock(&mutex);
            status = rankStatus[i];
            pthread_mutex_unlock(&mutex);

            if (!status)
            {
                continue;
            }

            int sendTask = NO_TASK;
            pthread_mutex_lock(&mutex);
            if (taskList.size() >= 2)
            {
                sendTask = taskList.back();
                taskList.pop_back();
            }
            pthread_mutex_unlock(&mutex);

            pthread_mutex_lock(&mutex);
            MPI_Allgather(&sendTask, 1, MPI_INT, newTasksList, 1, MPI_INT, MPI_COMM_WORLD);
            pthread_mutex_unlock(&mutex);

            int noTaskCounter = 0;
            for (int j = 0; j < size; ++j)
            {
                if (newTasksList[j] != NO_TASK && i == rank)
                {
                    pthread_mutex_lock(&mutex);
                    taskList.push_back(newTasksList[j]);
                    pthread_mutex_unlock(&mutex);
                }
                if (newTasksList[j] == NO_TASK)
                {
                    noTaskCounter++;
                }
            }
            if (noTaskCounter == size)
            {
                pthread_mutex_lock(&mutex);
                if (!workerFinish)
                {
                    iterationsFinished++;
                }
                workerFinish = true;
                pthread_mutex_unlock(&mutex);
            }
        }
    }
    pthread_exit(nullptr);
}


int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        MPI_Finalize();
        return 0;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    rankStatus = new bool[size];

    pthread_mutex_init(&mutex, nullptr);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    double start = MPI_Wtime();
    pthread_create(&receiverThread, &attr, receiver, NULL);
    pthread_create(&workerThread, &attr, worker, NULL);
    pthread_join(receiverThread, nullptr);
    pthread_join(workerThread, nullptr);
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&mutex);
    if (rank == 0)
    {
        std::cout << "Total statistics:" << std::endl;
        std::cout << "Average imbalance: " << summImbalance / totalIters * 100 << "%" << std::endl;
        std::cout << "Time spent on all iterations: " << MPI_Wtime() - start << " sec." << std::endl;
    }

    delete[] rankStatus;
    MPI_Finalize();
    return 0;
}