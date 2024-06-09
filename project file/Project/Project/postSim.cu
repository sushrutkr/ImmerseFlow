#include "postSim.cuh"
#include <cstdio>

// Postprocessing
// void write_results_to_file(const float* x, 
//                            const float* y, 
//                            const float* final_data,
//                            int ni, int nj,
//                            const char* filename){

//   FILE* fp = fopen(filename, "w"); // Open in write mode

//   if (fp == NULL) {
//     printf("Error opening file: %s\n", filename);
//     return;
//   }

//   fprintf(fp, "TITLE = \"Post Processing Tecplot\"\n");  // String
//   fprintf(fp, "VARIABLES = \"X\",\"Y\",\"T\"\n");  // String
//   fprintf(fp, "ZONE T=\"BIG ZONE\", I=%d, J=%d, DATAPACKING=POINT\n", ni, nj);  // Integers

//   // Write data in row-major order
//   for (int j = 0; j < nj; j++) {
//     for (int i = 0; i < ni; i++) {
//       fprintf(fp, "%f,%f,%f\n", x[i], y[j], final_data[i * nj + j]);
//     }
//   }

//   fclose(fp);
// }
