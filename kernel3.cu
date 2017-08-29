#include "definitions.cuh"
#include "assert.h"
#include <stdio.h>

__device__  void edgePerNet(int *in, int *out, int *srchAry, int numNodes, const int maxParents, int c){

	int sum = 0;
	for (int k = 1; k <= blockIdx.x; k++)
	{
		for (int i = 1; i <numNodes; i++)
		{
			int start = srchAry[i - 1];
			int stop = srchAry[i];
			int nodeSum = 0;
			for (int j = start; j < stop; j++)
			{
				if (in[j + ((k - 1) * c)] == 1)
				{
					nodeSum++;
				}
			}
			if (nodeSum > maxParents)
			{
				nodeSum = maxParents;
			}
			sum += nodeSum;
		}
	}


	out[blockIdx.x] = sum;

	/*int start = 0;
	int stop = blockID;
	int sum = 0;
	for (int i = start; i < stop; i++){
	if (in[i] == 1){
	sum++;
	}
	}*/

	out[blockIdx.x] = sum;
	/*JR*/
}

__device__ int nodeMaker(int c, int idx, int *data, int *srchAry){
	int sum = 0;
	//(threadIdx.x == 0 || threadIdx.x == 1)
	if (threadIdx.x == 0){
		return sum;
	}
	//shared memory code
	//else{ //srchAry[threadIdx.x-1]
	//	for (int i = 0; i < srchAry[threadIdx.x - 1]; i++){
	//		if (data[i] == 1){
	//			sum++;
	//		}
	//	}

	//	return sum;
	//}

	//normal non shared memory code
	else{ //srchAry[threadIdx.x-1]
		for (int i = 0; i < srchAry[threadIdx.x - 1]; i++){
			if (data[i + blockIdx.x * c] == 1){
				sum++;
			}
		}

		return sum;
	}
}

__device__ void edgeCounter(int c, int *edgesPN, int index, int *data, int *nodeAry, int *srchAry, int *edgeAry, const int maxParents){

	int place = (edgesPN[blockIdx.x] + nodeAry[index]);

	int pos = 0;
	int parentCounter = 0;

	int start = srchAry[threadIdx.x - 1];
	int stop = srchAry[threadIdx.x];
	//normal non shared memory code
	for (int i = start; i < stop; i++){
		if (parentCounter >= maxParents)
		{
			break;
		}
		if (data[blockIdx.x*c + i] == 1){
			edgeAry[place] = pos;
			place++;
			parentCounter++;

		}


		pos++;
	}
	//shared memory code
	/*
	for (int i = start; i < stop; i++){
	if (parentCounter >= maxParents)
	{
	break;
	}
	if (data[i] == 1){
	edgeAry[place] = pos;
	place++;
	parentCounter++;
	}
	pos++;
	}*/
	/*JR, GS*/
}

__device__ void nodeArrayParentCap(int numNodes, int *nodeArray, int *edgesPN, const int maxParents)
{

	//get starting place in node array to correct
	int start = blockIdx.x * blockDim.x;
	int stop = start + numNodes - 1;
	//only have one thread operate per block
	if (threadIdx.x == 0)
	{
		for (int i = start; i < stop; i++)
		{

			if ((nodeArray[i + 1] - nodeArray[i]) > maxParents)
			{
				int temp = nodeArray[i + 1];
				nodeArray[i + 1] = nodeArray[i] + maxParents;
				int decrement = temp - nodeArray[i + 1];
				//should this be <= or <???
				for (int j = i + 2; j <= stop; j++)
				{
					nodeArray[j] = nodeArray[j] - decrement;
				}
			}
		}
	}
}

__global__ void run22(int c, int *edgesPN, int *data, int *nodeAry, int noGenes, int noEdges, int *srchAry, int *edgeAry, const int maxParents){

	int index = threadIdx.x + blockDim.x * blockIdx.x;
	/*__shared__ int sharedData[45];
	if (threadIdx.x == 0)
	{
	memcpy(sharedData, &data[blockIdx.x * c], c * sizeof(int));
	}
	__syncthreads();*/

	nodeAry[index] = nodeMaker(c, index, data, srchAry);//-normal code
	//nodeAry[index] = nodeMaker(c, index, sharedData, srchAry); //-shared memory code

	__syncthreads();
	// oneCounter(c, threadIdx.x, data, nodeAry, placeHolder);

	nodeArrayParentCap(noGenes, nodeAry, edgesPN, maxParents);

	__syncthreads();

	if (threadIdx.x > 0){ edgeCounter(c, edgesPN, index, data, nodeAry, srchAry, edgeAry, maxParents); }//normal code
	//if (threadIdx.x > 0){ edgeCounter(c, edgesPN, index, data, nodeAry, srchAry, edgeAry); }
	//if (threadIdx.x > 0){ edgeCounter(c, edgesPN, index, sharedData, nodeAry, srchAry, edgeAry, maxParents); } //-shared memory code



	/*JR, GS*/
}

