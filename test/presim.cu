#include <iostream>
#include <vector>
#include <cstdio>

void write_results_to_file(int ni, int nj, std::vector<float>& final_data, const char* filename) {
    FILE* fp = fopen(filename, "w"); // Open in write mode

    if (fp == NULL) {
        printf("Error opening file: %s\n", filename);
        return;
    }

    fprintf(fp, "TITLE = \"Post Processing Tecplot\"\n");  // String
    fprintf(fp, "VARIABLES = \"T\"\n");  // String
    fprintf(fp, "ZONE T=\"BIG ZONE\", I=%d, J=%d, DATAPACKING=POINT\n", ni, nj);  // Integers

    // Write data in row-major order
    for (int j = 0; j < nj; j++) {
        for (int i = 0; i < ni; i++) {
            int index = i + j * ni;
            fprintf(fp, "%i,%i,%f\n", i, j, final_data[index]);
        }
    }

    fclose(fp);
}

int main() {
    // Example usage
    int ni = 3;
    int nj = 3;
    std::vector<float> final_data = {1.0, 2.0, 3.0,
                                      4.0, 5.0, 6.0,
                                      7.0, 8.0, 9.0};
    const char* filename = "output.txt";
    write_results_to_file(ni, nj, final_data, filename);

    return 0;
}
