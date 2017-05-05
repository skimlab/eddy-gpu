/*
functions.cu - CPU functions used in main()
includes kernel calls used to access the GPU
actual kernel code (run2, edgesPerNetworkKernel, run22, run25, run4) are all located in kernel.cu

*/



#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <math_functions.h>
#include <math.h>
#include <string.h>
#include <cstdlib>
#include <sys/timeb.h>
#include <assert.h>
#include <time.h>
#include <vector>
#include "definitions.cuh"
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )



static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg,
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void idPrep(int fixd, int combo, int *ary1, int *ary2){

	int start = 2;
	int pos = 1;
	int pos2 = 0;
	for (int i = 0; i <combo; i++){
		ary1[i] = pos;
		ary2[i] = pos2;
		pos++;
		if (pos == fixd){
			pos = start;
			pos2++;
			start++;
		}
	}


}
//cooley function 
double kool(double *P, double *Q, int scaler, int scaler1) {
	//	js = kool(lval1, sea, 0, scaler) / 2 + kool(lval1, sea, scaler, scaler) / 2;
	int numberOfValue = scaler1;
	double D = 0;
	for (int i = 0; i < numberOfValue; i++) {
		if (Q[i] == 0) {
			continue;
		}

		if (P[i + scaler] == 0)
			continue;

		double temp = P[i + scaler] * log(P[i + scaler] / Q[i]);
		D += temp;
	}

	return D;
}

// creates unique required data for subsequent steps
int structureUnique(int unique, int totEdges, int scaler1, int scaler2, int noGenes, bool *uniqueList, int *edgesPN, int *edgeAry, int *nodeAry, int *uniEdges, int *uniNodes, int *uniEpn){
	int start = 0;
	int start2 = 0;
	int set1 = 0;
	int count = 0;
	int count2 = 0;
	uniEpn[0] = 0;
	int place = 1;

	for (int i = 0; i < scaler2; i++){
		if (uniqueList[i] == 1){
			if (i < scaler1){
				set1++;
			}
			//compute number of Edges in network
			int numEdges = 0;
			if (i == scaler2 - 1){
				assert(i < scaler2 + 1);
				numEdges = totEdges - edgesPN[i];
			}
			else
			{
				assert(i < scaler2 + 1);
				numEdges = edgesPN[i + 1] - edgesPN[i];
			}
			//add matching values to new edge list
			int val;
			for (int j = 0; j < numEdges; j++){
				assert(i < scaler2 + 1);
				val = edgeAry[edgesPN[i] + j];
				uniEdges[j + start] = val;
				count++;
			}

			start = count;
			//add matching values to new node list
			for (int k = 0; k < noGenes; k++){
				uniNodes[k + start2] = nodeAry[i*noGenes + k];
				count2++;
			}
			start2 = count2;
			//add matching values to new master 
			assert(place < unique + 1);
			/*if (place >= unique)
			{
				printf("place : %d\n", place);
			}*/
			uniEpn[place] = numEdges + uniEpn[place - 1];
			place++;

		}
	}
	return set1;
}
//gamma function
double gammds(double x, double p, int *ifault)

//****************************************************************************80
{
	double a;
	double arg;
	double c;
	double e = 1.0E-09;
	double f;
	//int ifault2;
	double uflo = 1.0E-37;
	double value;
	//
	//  Check the input.
	//
	if (x <= 0.0)
	{
		*ifault = 1;
		value = 0.0;
		return value;
	}

	if (p <= 0.0)
	{
		*ifault = 1;
		value = 0.0;
		return value;
	}
	//
	//  LGAMMA is the natural logarithm of the gamma function.
	//
	arg = p * log(x) - lgamma(p + 1.0) - x;

	if (arg < log(uflo))
	{
		value = 0.0;
		*ifault = 2;
		return value;
	}

	f = exp(arg);

	if (f == 0.0)
	{
		value = 0.0;
		*ifault = 2;
		return value;
	}

	*ifault = 0;
	//
	//  Series begins.
	//
	c = 1.0;
	value = 1.0;
	a = p;

	for (;;)
	{
		a = a + 1.0;
		c = c * x / a;
		value = value + c;

		if (c <= e * value)
		{
			break;
		}
	}

	value = value * f;

	return value;
	/*JR*/
}




__host__ int cmpfunc(const void * a, const void * b)
{
	return (*(int*)a - *(int*)b);
}

__host__ void writeBdeuScores(char *outputFile, char *inputFile, char *classFile, char *genesetFile, char *class1, char *class2, int scaler, double *bdeuScores)
{
	FILE *scoreFile = fopen(outputFile, "w");
	if (scoreFile == NULL)	{ printf("scoreFile is NULL. Error code : %s. Exiting...\n", strerror(errno)); exit(EXIT_FAILURE); }
	fprintf(scoreFile, "\t\t BDEU SCORES\n");
	fprintf(scoreFile, "input : %s\n", inputFile);
	fprintf(scoreFile, "class : %s\n", classFile);
	fprintf(scoreFile, "geneset : %s\n\n", genesetFile);
	fprintf(scoreFile, "Network \t  %s  \t   \t    %s\n", class1, class2);
	int networkIndex = 1;
	for (int j = 0; j < scaler; j++)
	{
		fprintf(scoreFile, "Network %d\t %f \t %f \n", networkIndex++, bdeuScores[j], bdeuScores[j + scaler]);
		//printf("lval1[%d] : %f\n lval1", j, lval1[j]);
	}
	fclose(scoreFile);
	printf("Finished bdeu file %s\n", outputFile);
}

__host__ void writeNetworkFile(char *outputFile, char *inputFile, char *classFile, char *genesetFile, int networkNum, int *edgesPN,
	char genesetgenes[][40], int genesetlength, int *nodes, int *edges, int totalEdges, int *networkIds)
{
	FILE *output = fopen(outputFile, "w");
	if (output == NULL)	{ printf("output when accessing networkOutput.txt is NULL. Error code : %s. Exiting...\n", strerror(errno)); exit(EXIT_FAILURE); }

	fprintf(output, "%s", "\t\tUnique Networks\n\n");
	fprintf(output, "input : %s\n", inputFile);
	fprintf(output, "class : %s\n", classFile);
	fprintf(output, "geneset : %s\n\n\n", genesetFile);
	int edgePos = 0;

	//loop through all the unique networks
	for (int i = 0; i < networkNum; i++)
	{
		int total;
		if (i == networkNum - 1)
		{
			total = totalEdges - edgesPN[i];
		}
		else
		{
			total = edgesPN[i + 1] - edgesPN[i];
		}
		fprintf(output, "network %d (%d)\t number of edges : %d\n", i + 1, networkIds[i], total);
		fprintf(output, "%s", "---------------");
		fprintf(output, "%s", "---------------\n\n");
		if (total == 0)
		{
			continue;
		}
		//loop through each node in the ith network
		for (int j = 0; j < genesetlength - 1; j++)
		{
			//if change in node values --> edge has been found
			if (nodes[(i * genesetlength) + j + 1] != nodes[(i * genesetlength) + j])
			{
				int change = nodes[(i * genesetlength) + j + 1] - nodes[(i * genesetlength) + j];
				total -= change;
				for (int k = 0; k < change; k++)
				{
					fprintf(output, "%s - %s\n", genesetgenes[j], genesetgenes[edges[edgePos++]]);
					//fprintf(output, "%d - %d ---> normal case total remaining : %d j : %d\n", j, pUniEdges[edgePos++], total, j);
				}
			}
			if (j + 1 == genesetlength - 1 && total > 0)
			{
				//fprintf(output, "%s", "9s case\n");
				for (int k = 0; k < total; k++)
				{
					//fprintf(output, "%d - %d ---> 9s case\n", j + 1, pUniEdges[edgePos++]);
					fprintf(output, "%s - %s\n", genesetgenes[j + 1], genesetgenes[edges[edgePos++]]);
				}
			}



		}
		fprintf(output, "%s", "\n");
		fprintf(output, "%s", "\n");

	}

	fclose(output);
	printf("Finished writing %s\n", outputFile);
}

__host__ void writeEdgeListFile(char *outputFile, char *inputName, char *className, char *pathwayName, char genesetgenes[][40],
	int genesetlength, int *nodes, int *edges, int *edgesPN, int *priorMatrix, char class1[50], char class2[50])
{
	FILE *output = fopen(outputFile, "w");
	//arrays to keep track of found nodes w/edges for each class	
	std::vector<graphNode> class1Nodes;
	std::vector<graphNode> class2Nodes;


	//loop through *nodes/*edges to find get nodes and edge
	//store found nodes into either class1Nodes or class2Nodes with a graphNode struct
	int edgePos = 0;
	char classes[2][50];
	strcpy(classes[0], class1);
	strcpy(classes[1], class2);
	//loop through all the unique networks
	for (int i = 0; i < 2; i++)
	{
		int total;
		if (i == 2 - 1)
		{
			total = edgesPN[2] - edgesPN[i];
		}
		else
		{
			total = edgesPN[i + 1] - edgesPN[i];
		}
		if (total == 0)
		{
			continue;
		}
		//loop through each node in the ith network
		for (int j = 0; j < genesetlength - 1; j++)
		{
			//if change in node values --> edge has been found
			if (nodes[(i * genesetlength) + j + 1] != nodes[(i * genesetlength) + j])
			{
				int change = nodes[(i * genesetlength) + j + 1] - nodes[(i * genesetlength) + j];
				total -= change;
				for (int k = 0; k < change; k++)
				{
					graphNode temp;
                                        temp.node = j;
                                        temp.edge = edges[edgePos++];
					temp.inBoth = 0;
					//fprintf(output, "%s - %s\n", genesetgenes[j], genesetgenes[edges[edgePos++]]);
					//printf("%s\t%s\t%s\n", genesetgenes[j], genesetgenes[edges[edgePos++]], classes[i]);
					//printf("%s\n", classes[i]);
					if(strcmp(classes[i], class1) == 0)
					{
						class1Nodes.push_back(temp);
					}
					else
					{
						class2Nodes.push_back(temp);
					}
					//fprintf(output, "%d - %d ---> normal case total remaining : %d j : %d\n", j, pUniEdges[edgePos++], total, j);
				}
			}
			if (j + 1 == genesetlength - 1 && total > 0)
			{
				for (int k = 0; k < total; k++)
				{
					graphNode temp;
                                        temp.node = j + 1;
                                        temp.edge = edges[edgePos++];
					temp.inBoth = 0;
					//fprintf(output, "%d - %d ---> 9s case\n", j + 1, pUniEdges[edgePos++]);
					//fprintf(output, "%s - %s\n", genesetgenes[j + 1], genesetgenes[edges[edgePos++]]);
					//printf("%s\t%s\t%s\n", genesetgenes[j + 1], genesetgenes[edges[edgePos++]], classes[i]);
					//printf("%s\n", classes[i]);
                                        if(strcmp(classes[i], class1) == 0)
                                        {
                                                class1Nodes.push_back(temp);
                                        }
                                        else
                                        {
                                                class2Nodes.push_back(temp);
                                        }
				}
			}
		}
	}


	//determine which nodes in class1 are also in class2 - if the are mark inBoth for each node so later we dont repeat when we print
	for(int i = 0; i < class1Nodes.size(); i++)
	{
		for(int j = 0; j < class2Nodes.size(); j++)
		{
			if(class1Nodes.at(i).node == class2Nodes.at(j).node && class1Nodes.at(i).edge == class2Nodes.at(j).edge)
			{
				class1Nodes.at(i).inBoth = 1;
				class2Nodes.at(j).inBoth = 1;
				
			}
		}
	}

	//create spacer and ff rows to offset our reads for row/col access to the prior matrix
 	int c = ((genesetlength * genesetlength) - genesetlength) / 2;
	int position = 0;
        int *spacer = (int *)malloc(sizeof(int) * c);
        int *ff = (int *)malloc(sizeof(int) * c);
        for (int row = 1; row < genesetlength; row++)                     
        {
                for (int col = 0; col < row; col++)
                {
                        spacer[position] = row;
                        ff[position] = col;
                        position++;
                }
        }

	//denote if eddy found an edge so that we can check later if the edge wasnt found- 1 = EDDY found edge 0 = EDDY  did not find edge
	int *eddyFound = (int *)calloc(genesetlength * genesetlength, sizeof(int));
	



	
	char priorString[2][20] = {"NONE", "PRIOR"};
	//printf("ATTEMPTING FILE OUTPUT TEST: \n");
	for(int i = 0; i < class1Nodes.size(); i++)
	{
		
		int row = class1Nodes.at(i).node;
                int col = class1Nodes.at(i).edge;
		if(class1Nodes.at(i).inBoth == 1)
		{
			//if node is in both classes print to file with "Both" class specifier
			fprintf(output, "%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col], "Both", priorString[*(priorMatrix + row * genesetlength + col)]);
			//printf("%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col], "Both", priorString[*(priorMatrix + row * genesetlength + col)]);
			//denote that eddy found this edge
			*(eddyFound + row * genesetlength + col) = 1;
			*(eddyFound + col * genesetlength + row) = 1;
			//printf("IN BOTH!");
		}
		else
		{
			//if it isnt both class write all class1 nodes
			fprintf(output, "%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col], class1, priorString[*(priorMatrix + row * genesetlength + col)]);
			//denote that eddy found this edge
			*(eddyFound + row * genesetlength + col) = 1;
                        *(eddyFound + col * genesetlength + row) = 1;
		}
	}
	
	for(int i = 0; i < class2Nodes.size(); i++)
	{
		int row = class2Nodes.at(i).node;
                int col = class2Nodes.at(i).edge;
		if(class2Nodes.at(i).inBoth != 1)
		{
			//write to file all nodes as long as not also in class1 - that way we dont repeat edges
			fprintf(output, "%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col], class2, priorString[*(priorMatrix + row * genesetlength + col)]);
			//printf("%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col], class2, priorString[*(priorMatrix + row * genesetlength + col)]);
			//denote that eddy found an edge
			*(eddyFound + row * genesetlength + col) = 1;
                        *(eddyFound + col * genesetlength + row) = 1;
		}
	}
	

	//write edges that werent found by EDDY
	//loop through prior matrix- if relationship isnt already found by eddy but in the prior file, write relationship to file
	for(int i = 0; i < c; i++)
	{
		int row = spacer[i];
		int col = ff[i];
		//0 in eddyFound = eddy did not find 1 in eddyFound = eddy did find
		if(*(eddyFound + row * genesetlength + col) == 0 && *(priorMatrix + row * genesetlength + col) == 1)
		{
			fprintf(output,"%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col], "Neither", "PRIOR");
		}
	}
	
	free(spacer); spacer = NULL;
	free(ff); ff = NULL;
	free(eddyFound); eddyFound = NULL;
	
	fclose(output);
	printf("Finished writing edgelist file %s\n", outputFile);

}

__host__ double mean(double *a, size_t size)
{
	double sum = 0.0;
	for (size_t i = 0; i < size; i++)	{ sum += a[i];	}
	return sum / size;
}

__host__ double variance(double mean, double *a, size_t size)
{
	double sum = 0.0;
	for (size_t i = 0; i < size; i++)	{ sum += pow(a[i] - mean, 2); }
	return sum / size;
}


__host__ void checkParentLimit(int numNetworks, int numNodes, int maxParents, int *nodes, size_t size)
{
	//loop through each network
	for (int net = 0; net < numNetworks; net++)
	{
		//loop through nodes in each network
		for (int i = 0; i < numNodes - 1; i++)
		{
			if (nodes[(net * numNodes) + i + 1] - nodes[(net * numNodes) + i] > maxParents)
			{
				printf("PROBLEM @ %d %d\n", (net * numNodes) + i, (net * numNodes) + i + 1);
			}
		}
	}
}

//compute likelihood of different dataset parsed by 2 iterations
__host__ void compute_likelihood(int scaler, int noNodes, double *out5, double *lval1)
{
	for(int g = 0; g < 2; g++){
		int set = 0;
		int place = 0;
		double scoreSum = 0;
		double *likeli1;
		double min = 0;
		double max = 0;
		double inAlpha = 0;
		double probScale = 0;
		double likeliSum = 0;
		double nonInf = 0;
		double *dist;

		double *adjusted;
		double *infFlag;
		double *outq;
		int localoffset;
		outq = out5;
		if (g < 1){
			localoffset = 0;
		}
		else
		{
			localoffset = scaler;
		}

		dist = (double *)malloc(sizeof(double)*scaler);
		likeli1 = (double *)malloc(sizeof(double)*scaler);
		adjusted = (double *)malloc(sizeof(double)*scaler);
		infFlag = (double *)malloc(sizeof(double)*scaler);
		for (int k2 = 0; k2 < scaler; k2++){

			dist[k2] = 0.0;

			adjusted[k2] = 0.0;
			infFlag[k2] = 0.0;

		}
		for (int i = 0; i < scaler*noNodes; i++){

			dist[place] += outq[i + localoffset*noNodes];
			set++;
			if (set == noNodes){
				set = 0;
				place++;
			}
		}

		min = dist[0];
		max = dist[0];
		for (int j3 = 0; j3 < scaler; j3++){
					//			printf(" dis %d %f \n", j3, dist[j3]);
			scoreSum += dist[j3];
			if (dist[j3]>max){

				max = dist[j3];
			}
			if (dist[j3] < min){

				min = dist[j3];
			}

		}


		inAlpha = -1 * (scoreSum / scaler);
		//printf("\n min-%f max-%f", min, max);
		//printf("inAlpha-%f", inAlpha);
		probScale = (10) / (max - min);

		for (int m = 0; m < scaler; m++){

			adjusted[m] = (dist[m] + inAlpha)*probScale;
			//			printf("\n adjusted: %f", adjusted[m]);
			likeli1[m] = exp(adjusted[m]);
			//			printf("\n likeli: %f", likeli1[m]);
			nonInf += likeli1[m];
			//likeLi[m] = posInf;
			//suppress overflow infinity error
			#pragma warning(suppress: 4056)
			if (likeli1[m] >= INFINITY || likeli1[m] <= -INFINITY){
				infFlag[m] = 1.0;
				likeliSum++;
			}

		}
		free(dist); dist = NULL;
		free(adjusted); adjusted = NULL;
		//	printf("\n likesum: %f nonInf: %f", likeliSum, nonInf);

		if (likeliSum == 0){
			for (int meow = 0; meow < scaler; meow++){
				likeli1[meow] = likeli1[meow] / nonInf;
				lval1[meow + localoffset] = likeli1[meow];

			}

		}
		else{
			for (int meow = 0; meow < scaler; meow++){
				likeli1[meow] = infFlag[meow] / likeliSum;
				lval1[meow + localoffset] = likeli1[meow];
			}
		}
		free(likeli1); likeli1 = NULL;
		free(infFlag); infFlag = NULL;
		outq = NULL;
	}

}

//determine gene relationships by running expression data through chi-squared test
//@RETURN: sampleSum - total number of samples 
int calculate_edges(int scalerSum, int samples, int samples2, int genesetlength, int size2, int c, int genes, int *transferdata1, int *transferdata2, int *ff1, int *priorMatrix, int *spacer1, cudaEvent_t start, cudaEvent_t stop, float time, double pw, int *edgeListData1, int *edgeListData2, int *dout23, int *dedgesPN, int *dtriA, int *dtriAb, int *ddofout, int *dppn, int *dstf, int *dff, int *dspacr, int *dpriorMatrix, int numclass1, int numclass2, int permNum)
{
			

	//copy into device 
	assert(genes*samples*sizeof(int) == genesetlength * numclass1 * sizeof(int));
	assert(genes*samples2*sizeof(int) == genesetlength * numclass2 * sizeof(int));
	HANDLE_ERROR(cudaMemcpy(dtriA, transferdata1, genes*samples*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dtriAb, transferdata2, genes*samples2*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dff, ff1, c*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dspacr, spacer1, c*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dpriorMatrix, priorMatrix, genesetlength * genesetlength * sizeof(int), cudaMemcpyHostToDevice));
			
			

	//no longer used once copied to GPU
	free(spacer1); spacer1 = NULL;
	free(ff1); ff1 = NULL;

	int sampleSum = samples + samples2 + 2;

			
	cudaEventRecord(start, 0);
	if( c < MAX_THREADS)
	{
	run2 << <sampleSum, c, genes * genes * sizeof(int) >> >(genes, samples, samples2, dtriA, dtriAb, dspacr, dff, ddofout, dppn, dstf, dout23, c, dpriorMatrix, pw);
	}
	else
	{
		int BPN = ceil((c * 1.0) / MAX_THREADS);
		int TPB = ceil((c * 1.0) / BPN);
			
		run2Scalable <<< sampleSum * BPN, TPB>>>(genes, samples, samples2, dtriA, dtriAb,dspacr, dff, ddofout, dppn, dstf, dout23, c, dpriorMatrix, pw, BPN, TPB);
	}



	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	if (permNum == 0)
	{
		//holds post run2 edge data for edge list calculations after permutations
		edgeListData1 = (int *)malloc(sizeof(int) * c);
		edgeListData2 = (int *)malloc(sizeof(int) * c);

		//host array to transfer output of run2 to edgeListData1/edgeListData2
		int *tempOut23 = (int *)malloc(sizeof(int) * c * scalerSum);
				

		//copy binary data back to CPU
		HANDLE_ERROR(cudaMemcpy(tempOut23, dout23, sizeof(int) * c * scalerSum, cudaMemcpyDeviceToHost));
				

		//first network in first class - no samples left out
		for (int i = 0; i < c; i++)
		{
			//load 1st network from class 1
			edgeListData1[i] = tempOut23[i];
		}
		int count = 0;
		//last network in 2nd class - no samples left out
		for (int i = (scalerSum - 1) * c; i < (scalerSum) * c; i++)
		{
			edgeListData2[count++] = tempOut23[i];
		}
			



		free(tempOut23); tempOut23 = NULL;
	}



	return sampleSum;
}

//build edgesPN array that holds the number of edges for each network
void count_edges(int *dsrchAry, cudaEvent_t PN_start, cudaEvent_t PN_stop, float PN_time, int *dout23, int *dedgesPN, int genes, int MAX_PARENTS, int c, int scalerSum, int *edgesPN, cudaError_t errSync, int sampleSum)
{

	int *tempEdgesSums;
	tempEdgesSums = (int *) calloc(sampleSum + 1, sizeof(int));
	edgePerNetworkKernel << < sampleSum + 1, 1 >> > (dout23, dedgesPN, dsrchAry, genes, MAX_PARENTS, c);
	cudaEventRecord(PN_stop, 0);
			
	//copy edge sums over to CPU to calculate prefix sum for edgesPN	
	HANDLE_ERROR(cudaMemcpy(tempEdgesSums, dedgesPN, sizeof(int) * (scalerSum + 1), cudaMemcpyDeviceToHost));	
			
	edgesPN[0] = 0;
	for(int i = 1; i < sampleSum + 1; i++)
	{
		edgesPN[i] = edgesPN[i-1] + tempEdgesSums[i-1]; //prefix sum calculation
	}
	//get rid of this temp array	
	free(tempEdgesSums); tempEdgesSums = NULL;
	//edgesPN on the CPU is now fixed but dedgesPN is used later- copy edgesPN results back to GPU memory
	HANDLE_ERROR(cudaMemcpy(dedgesPN, edgesPN, sizeof(int) * (sampleSum + 1), cudaMemcpyHostToDevice));
			
			
	errSync = cudaGetLastError();
	if (errSync != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(errSync));
	}
	cudaEventSynchronize(PN_stop);
	cudaEventElapsedTime(&PN_time, PN_start, PN_stop);

}

//construct network graphs from gene relationships and network edge counts (edgesPN).
void build_graph(int scalerSum, int noNodes, int c, int *dedgesPN, int *dout23, int *dpNodes, int numEdges, int *dsrchAry, int *dpEdges, int MAX_PARENTS, int *pNodes, int *pEdges, int nodeSize, int edgeSize)
{
	run22 << <scalerSum, noNodes >> >(c, dedgesPN, dout23, dpNodes, noNodes, numEdges, dsrchAry, dpEdges, MAX_PARENTS);

	HANDLE_ERROR(cudaMemcpy(pNodes, dpNodes, nodeSize, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pEdges, dpEdges, edgeSize, cudaMemcpyDeviceToHost));

			

	//ensure parent limit
	checkParentLimit(scalerSum, noNodes, MAX_PARENTS, pNodes, nodeSize / sizeof(int));

}




//determine which graphs are duplicated and mark them for removal in structureUnique
void mark_duplicate_networks(int scalerSum, int scalerCombo, int *scalerTest, int *shrunkPlc, int *shrunk, int *dshrunk, int *dscalerTest, int *dshnkplc, int maxThreads, int samples, int numEdges, int genesetlength, int *dedgesPN, int *dpNodes, int *dpEdges)
{

	idPrep(scalerSum, scalerCombo, scalerTest, shrunkPlc);

	

	//cp into device
	HANDLE_ERROR(cudaMemcpy(dscalerTest, scalerTest, sizeof(int)*scalerCombo, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dshnkplc, shrunkPlc, sizeof(int)*scalerCombo, cudaMemcpyHostToDevice));

	run25 << <(scalerCombo / (maxThreads - 1)) + 1, maxThreads - 1 >> >(samples + 1, scalerSum, numEdges, genesetlength, scalerCombo, dedgesPN, dpNodes, dpEdges, dshrunk, dscalerTest, dshnkplc);
	
	HANDLE_ERROR(cudaMemcpy(shrunk, dshrunk, sizeof(int)*scalerCombo, cudaMemcpyDeviceToHost));
	cudaFree(dshrunk); dshrunk = NULL;
	cudaFree(dscalerTest); dscalerTest = NULL;
	cudaFree(dshnkplc); dshnkplc = NULL;
	cudaFree(dedgesPN); dedgesPN = NULL;
	cudaFree(dpEdges); dpEdges = NULL;
	cudaFree(dpNodes); dpNodes = NULL;
	free(shrunkPlc); shrunkPlc = NULL;


}
//calculate the BDEU scores for each graph
void score_networks(double *out5, double *dout5, int *dpEdges2, int *dpNodes2, int *dNij, int *dNijk, int *dUniEpn, int uniEdgeSize, int uniNodeSize, int unisum, int noNodes, int scaler, int *pUniEdges, int *pUniNodes, int *pUniEpn, int genesetlength, int edSum, int samples, int samples2, int *dtriA, int *dtriAb, int *dppn, int *dstf, int dppnLength)
{


	

	HANDLE_ERROR(cudaMemcpy(dpEdges2, pUniEdges, uniEdgeSize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dpNodes2, pUniNodes, uniNodeSize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dUniEpn, pUniEpn, sizeof(int) * unisum, cudaMemcpyHostToDevice));
	free(pUniNodes); pUniNodes = NULL;
	free(pUniEdges); pUniEdges = NULL;
	free(pUniEpn); pUniEpn = NULL;

	cudaEvent_t run4Start, run4End;
	cudaEventCreate(&run4Start);
	cudaEventCreate(&run4End);
	cudaEventRecord(run4Start, 0);
	float run4Time;

	run4 << <scaler * 2, noNodes >> >(scaler, dUniEpn, genesetlength, edSum, unisum, samples, samples2, dtriA, dtriAb, dpEdges2, dpNodes2, dppn, dstf, dNij, dNijk, dout5, dppnLength);
	
	cudaEventRecord(run4End, 0);
	cudaEventSynchronize(run4End);
	cudaEventElapsedTime(&run4Time, run4Start, run4End);
	HANDLE_ERROR(cudaMemcpy(out5, dout5, sizeof(double)*noNodes*scaler * 2, cudaMemcpyDeviceToHost));


}
