#include "nfft3.h"
#include <vtk/vtksys/Configure.h>
#include <vtk/vtksys/Configure.hxx>
#include <vtk/vtkSmartPointer.h>
#include <vtk/vtkFloatArray.h>
#include <vtk/vtkDoubleArray.h>
#include <vtk/vtkPolyData.h>
#include <vtk/vtkInformation.h>
#include <vtk/vtkTable.h>
#include <vtk/vtkDelimitedTextWriter.h>
#include <vtk/vtkZLibDataCompressor.h>
#include <vtk/vtkXMLImageDataWriter.h>
#include <vtk/vtkXMLPolyDataWriter.h>
#include <vtk/vtkImageData.h>
#include <vtk/vtkPointData.h>

int main(void)
{
    int N1 = 16;
    int N2 = 16;
    int N3 = 16;
    int maxcells = 32;
    int ncomponents = 1;
    int n_space_div[] = {N1, N2, N3};
    float dd[] = {1.0f / (float)N1, 1.0f / (float)N2, 1.0f / (float)N3};
    float posl[] = {-0.5, -0.5, -0.5};
    int N[] = {N1, N2, N3};
    int M = N1 * N2 * N3;
    nfftf_plan p;

    nfftf_init(&p, 3, N, M); //(plan,3D,{i,j,k},i*j*k)
    for (int k = 0; k < N3; ++k)
        for (int j = 0; j < N2; ++j)
            for (int i = 0; i < N1; ++i)
            {
                int n = (k * N2 + j) * N1 + i;
                //int n = (i * N2 + j) * N1 + k;
                p.x[3 * n] = -0.5 + (float)i / N1; //p.x[n][0]=x ..,y,z
                p.x[3 * n + 1] = -0.5 + (float)j / N2;
                p.x[3 * n + 2] = -0.5 + (float)k / N3;
                if (i == N1 / 2)
                    p.f[n][0] = 1.0; /* Memory corruption? */
                else
                    p.f[n][0] = 0.0;
                p.f[n][1] = 0.0;
            }
    int xi = (n_space_div[0] - 1) / maxcells + 1;
    int yj = (n_space_div[1] - 1) / maxcells + 1;
    int zk = (n_space_div[2] - 1) / maxcells + 1;
    int nx = n_space_div[0] / xi;
    int ny = n_space_div[1] / yj;
    int nz = n_space_div[2] / zk;
    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New(); // Create the vtkImageData object
    imageData->SetDimensions(nx, ny, nz);                                           // Set the dimensions of the image data
    imageData->SetSpacing(dd[0] * xi, dd[1] * yj, dd[2] * zk);
    imageData->SetOrigin(posl[0], posl[1], posl[2]); // Set the origin of the image data
    imageData->AllocateScalars(VTK_FLOAT, 2);
    imageData->GetPointData()->GetScalars()->SetName("f_hat");
    float *data2 = static_cast<float *>(imageData->GetScalarPointer()); // Get a pointer to the density field array

    for (int k = 0; k < nz; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                int n = (k * ny + j) * nx + i;
                 for (int c = 0; c < 2; ++c)
                data2[2*n+c] = p.f[n][0];
            }
        }
    }

    vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New(); // Create the vtkXMLImageDataWriter object
    writer->SetFileName("f.vti");                                                             // Set the output file name                                                                     // Set the time value
    writer->SetDataModeToBinary();
    // writer->SetCompressorTypeToLZ4();
    writer->SetCompressorTypeToZLib(); // Enable compression
    writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
    writer->SetInputData(imageData);   // Set the input image data
    writer->Write();                   // Write the output file
    if (p.flags & PRE_ONE_PSI)
        nfftf_precompute_one_psi(&p);

    const char *check_error_msg = nfftf_check(&p);
    if (check_error_msg != 0)
    {
        printf("Invalid nfft parameter: %s\n", check_error_msg);
        return 1;
    }

    nfftf_adjoint(&p); /* Segfault if M <= 6 */
    nfftf_vpr_complex(p.f_hat, p.N_total, "adjoint nfft, vector f_hat");
    nfftf_finalize(&p); /* Segfault if M > 6 */




    imageData->GetPointData()->GetScalars()->SetName("f_hat");
  //  float *data2 = static_cast<float *>(imageData->GetScalarPointer()); // Get a pointer to the density field array

    for (int k = 0; k < nz; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                int n = (k * ny + j) * nx + i;
                 for (int c = 0; c < 2; ++c)
                //                data2[n] = p.f_hat[n][0];
                data2[2*n+c] = p.f_hat[n][0];
            }
        }
    }

     writer->SetFileName("f_hat.vti");                                                             // Set the output file name                                                                     // Set the time value

  //  writer->SetInputData(imageData);   // Set the input image data
    writer->Write();                   // Write the output file

    return 0;
}