#ifndef _FIRDirectII_H_
#define _FIRDirectII_H_

/************************************************************************/
/* Example                                                              */
/************************************************************************/

__device__ __constant__ float coeff_Kernel[KERNELTAPS];	//coeff parameters are placed in constant memory.

__global__ void calcFIR(const float * g_indata, float * g_outdata, const int CalcSize)
{
    __shared__ float shared[DATAPERCYCLE+THREAD_NUM];

	// access Block Width
	const unsigned int bw = gridDim.x;
	// access Block ID
	const unsigned int bix = blockIdx.x;

	// access thread id
	const unsigned int tid = threadIdx.x;

	float dOut;

	//do FIR
	//each threads has offseted address to global memory. loop jumps threads*blocks.
	for (int index = 0; index < CalcSize; index = index + THREAD_NUM*bw)
	{
		dOut = 0.0;

		//read g_indata to Shared Memory
		//cycle is, ex, 8=8192/1024.

		for (int j = 0; j < KERNELTAPS/DATAPERCYCLE; j++)
		{
			shared[tid             ] = g_indata[DATAPERCYCLE*j + THREAD_NUM*bix + index + tid               ];
			__syncthreads();
			shared[tid+THREAD_NUM  ] = g_indata[DATAPERCYCLE*j + THREAD_NUM*bix + index + tid + THREAD_NUM  ];
			__syncthreads();
			shared[tid+THREAD_NUM*2] = g_indata[DATAPERCYCLE*j + THREAD_NUM*bix + index + tid + THREAD_NUM*2];
			__syncthreads();

#pragma unroll 
			for(int k = 0; k < DATAPERCYCLE; k = k+1)
			{
				dOut += shared[k + tid] * coeff_Kernel[j*DATAPERCYCLE + k];
			}
		}
		__syncthreads();
		g_outdata[THREAD_NUM*bix + index + tid] = dOut;
	}
}

#endif