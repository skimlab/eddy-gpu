#include "definitions.cuh"
#include "assert.h"
#include <stdio.h>

__device__ void space4(int scaler, int index, int noEdges, int *edgesPN, int *nodeAry, int *temp, int *edgeAry){
	int sum = 0;
	int find = threadIdx.x;
	int start = edgesPN[blockIdx.x];
	int stop = edgesPN[blockIdx.x + 1];
	if (blockIdx.x == scaler - 1){ //22251244
		stop = noEdges;
	}


	for (int i = start; i < stop; i++){
		if (edgeAry[i] == find){
			sum++;
		}
	}
	temp[index] = sum;


	/*JR*/
}


__device__ void reshape(int idblock, int noNodes, int *temp, int *out2){
	int count = 0;
	int start = idblock*noNodes;
	for (int j = start + 1; j < start + noNodes; j++){
		count = temp[j - 1] + count;
		out2[j] = count;
	}
	out2[start] = 0;

}
//creates second place holder array before final parent array
__device__ void parentTrap(int scaler, int idblock, int index, int noEdges, int *edgesPN, int *out2, int *edgeAry, int *temp2){
	int finder = threadIdx.x;
	int incep = out2[index] + edgesPN[blockIdx.x];
	if (blockIdx.x == 0){
		//	incep = out2[finder];
	}
	int start = edgesPN[blockIdx.x];
	int stop = edgesPN[blockIdx.x + 1];
	if (blockIdx.x == scaler - 1){
		stop = noEdges;
	}


	for (int i = start; i < stop; i++){


		if (edgeAry[i] == finder){

			temp2[incep] = i;
			incep++;
		}


	}
}

__device__ void parentTrapPartduex(int scaler, int noEdges, int *edgesPN, int index, int noNodes, int *placeHolder, int *temp2, int *out2, int *out3){
	int start = out2[index] + edgesPN[blockIdx.x];
	int start2 = blockIdx.x*blockDim.x;
	//int posr = 1;

	int stop = out2[index + 1] + edgesPN[blockIdx.x];

	if (threadIdx.x == noNodes - 1){
		stop = edgesPN[blockIdx.x + 1];
	}
	if (blockIdx.x == scaler - 1){
		stop = noEdges;
	}
	for (int i = start; i < stop; i++)
	{
		for (int j = start2; j < start2 + noNodes; j++)
		{
			if (temp2[i] >= placeHolder[j] + edgesPN[blockIdx.x] && temp2[i] <placeHolder[j + 1] + edgesPN[blockIdx.x])
			{

				out3[i] = j - blockDim.x*blockIdx.x;

			}

		}


	}


}


__global__ void run3(int scaler, int *dedgesPN, int noEdges, int noNodes, int *temp, int *temp2, int *placeHolder, int *nodeAry, int *edgeAry, int *out2, int *out3){
	int index = threadIdx.x + blockDim.x*blockIdx.x;


	space4(scaler, index, noEdges, dedgesPN, nodeAry, temp, edgeAry);

	if (index < noEdges && index>-1){
		temp2[index] = 0;
	}
	__syncthreads();
	reshape(blockIdx.x, noNodes, temp, out2);



	//out3[index] = 0;
	//out2[index] = 0;
	__syncthreads();

	parentTrap(scaler, blockIdx.x, index, noEdges, dedgesPN, out2, edgeAry, temp2);

	__syncthreads();

	//	__syncthreads();

	parentTrapPartduex(scaler, noEdges, dedgesPN, index, noNodes, placeHolder, temp2, out2, out3);

	__syncthreads();

}

__global__ void edgePerNetworkKernel(int *input, int *output, int *srchAry, int numNodes, const int maxParents, int c)
{

	//edgePerNet(input, output, srchAry, numNodes, maxParents, c);

	//compute sum of edges for a single network and store in output[] - prefix sum is completed on CPU	
	
	int sum = 0;
	for(int i = 1; i < numNodes; i++)
	{
		int start = srchAry[i-1];
		int stop = srchAry[i];
		int nodeSum = 0;
		
		for(int j = start; j < stop; j++)
		{
			if(input[j + (blockIdx.x * c)] == 1)
			{
				nodeSum++;
			}
		}
	
		if(nodeSum > maxParents)
		{
			nodeSum = maxParents;
		}
		
		sum += nodeSum;
	}

	output[blockIdx.x] = sum;







	/*
	extern __shared__ int dataMem[];
	__shared__ int searcher[sizeof(int) * MAX_GENES];
	memcpy(searcher, srchAry, sizeof(int) * numNodes);
	int sum = 0;
	for (int k = 1; k <= blockIdx.x; k++)
	{
		//sync threads to prevent race condition
		//some threads finish early and overwrite dataMem with next data chunk
		__syncthreads();
		dataMem[threadIdx.x] = input[((k - 1) * c) + threadIdx.x];
		__syncthreads();
		for (int i = 1; i <numNodes; i++)
		{
			//int start = srchAry[i - 1];
			//int stop = srchAry[i];
			int start = searcher[i - 1];
			int stop = searcher[i];
			int nodeSum = 0;
			for (int j = start; j < stop; j++)
			{
				//if (input[j + ((k - 1) * c)] == 1)
				if (dataMem[j] == 1)
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
	output[blockIdx.x] = sum;
	*/

}

__global__ void noStates_kernel(const int noGenes, int samples1, int samples2, int *data1, int *data2, int *out1, int *out2)
{
int *dataIn;
	int samplesIn;
	int start;
	int stop;
	int retVal;


	if (blockIdx.x < noGenes){
		dataIn = data1;
		samplesIn = samples1;
		start = blockIdx.x*samplesIn;
		stop = start + samplesIn;
		assert(stop <= noGenes * samples1);
	}
	else{
		dataIn = data2;
		samplesIn = samples2;
		start = (blockIdx.x - noGenes)*samplesIn;
		stop = start + samplesIn;
		assert(stop <= noGenes * samples2);
	}



	unsigned short statedata1[3] = { 0, 0, 0 };

	for (int i = start; i < stop; i++)
	{
	
		if (dataIn[i] == -1){
			statedata1[0]++;
		}
		if (dataIn[i] == 0){
			statedata1[1]++;

		}
		if (dataIn[i] == 1){
			statedata1[2]++;

		}

	}


	retVal = 3;
	for (int i = 0; i < 3; i++){
		out2[blockIdx.x * 3 + i] = 1;
		if (statedata1[i] == 0){
			out2[blockIdx.x * 3 + i] = 0;

			retVal--;
		}




	}
	
	out1[blockIdx.x] = retVal;
}

