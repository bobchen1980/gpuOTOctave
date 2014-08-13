/********************************************************************
*  CUDA_OTOctave.cu
*  This is a main entrance of the CUDA program.
*********************************************************************/
// includes, system
#include <stdio.h>
#include <stdlib.h>
// includes, project
#include <cuda_runtime.h>
#include <cutil.h>

//define constant
#define KERNELTAPS	8192	//must be odd value. Freq/Taps = Filter Frequency accuracy. 5.4Hz for 44100.
#define THREAD_NUM  512		//executed thread count per block, do not change. shared memory is common in the block.
#define DATAPERCYCLE 8192	//data count per loop. do not change

// includes, kernels
#include "CUDA_InitCuda.cuh"
#include "CUDA_FIRDirectII.cuh"
/************************************************************************/
/* Main entrance                                                            */
/************************************************************************/
int main(int argc, char* argv[])
{

	if(!InitCUDA()) {
		return 0;
	}
	
	int packlen =  1024*16;

	float * h_idata;
	cudaMallocHost((void**)&h_idata, sizeof(float)*packlen);
	for(int i=0;i<packlen;i++)
		h_idata[i] = i;

	//allocate GPU device input memory
	float * d_idata;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_idata, sizeof(float)*packlen) );
	// allocate GPU device memory for result
	float * d_odata;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_odata, sizeof(float)*packlen) );
	float * h_odata;
	cudaMallocHost((void**)&h_odata, sizeof(float)*packlen);
	
	float FIRCoeff[KERNELTAPS];
	for(int i=0;i<KERNELTAPS;i++)
		FIRCoeff[i] = 0.0001;

				CUDA_SAFE_CALL( cudaMemcpy(d_idata, h_idata,sizeof(float)*packlen,cudaMemcpyHostToDevice));
				//copy coeffs to constant
				CUDA_SAFE_CALL( cudaMemcpyToSymbol(coeff_Kernel, FIRCoeff, sizeof(float)*KERNELTAPS) );


	unsigned int timer = 0;
	CUT_SAFE_CALL( cutCreateTimer( &timer));
	CUT_SAFE_CALL( cutStartTimer( timer));

	calcFIR<<<1, 512, 0>>>(d_idata, d_odata, packlen);
	CUT_CHECK_ERROR("Kernel execution failed\n");

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf("Kernel time: %f (ms)\n", cutGetTimerValue( timer));
	CUT_SAFE_CALL( cutDeleteTimer( timer));


				CUDA_SAFE_CALL( cudaMemcpy( (void *)h_odata, d_odata, sizeof(float)*packlen, cudaMemcpyDeviceToHost) );
			FILE *fo;
			fo = fopen("out.txt", "wb");
			for(int i=0;i<packlen;i++)
			{
				fprintf(fo,"%f ",h_odata[i]); 
				fprintf(fo,"\n"); 
			}
			fclose(fo);

	CUDA_SAFE_CALL( cudaFree(d_idata));
	CUDA_SAFE_CALL( cudaFree(d_odata));
	CUDA_SAFE_CALL( cudaFreeHost(h_odata));
	CUDA_SAFE_CALL( cudaFreeHost(h_idata));


	CUT_EXIT(argc, argv);

	return 0;
}
