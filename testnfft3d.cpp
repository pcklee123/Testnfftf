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
    int N1 = 32; // x
    int N2 = 64; // y
    int N3 = 128; // z
                 //    int maxcells = 32;
                 //    int ncomponents = 3;
    //  int n_space_div[] = {N3, N2, N1};
    double dd[] = {1.0, 1.0, 1.0};
    double posl[] = {-(double)N1 / 2, -(double)N2 / 2, -(double)N3 / 2};
    NFFT_INT Nn[] = {N1, N2, N3};
    int N[] = {N1, N2, N3};
    int M = N1 * N2 * N3;
    nfftf_plan p;

    nfftf_init(&p, 3, N, M);
    for (int k = 0; k < N3; ++k)
        for (int j = 0; j < N2; ++j)
            for (int i = 0; i < N1; ++i) // i is x-coord and array is [k][j][i]
            {
                int n = (k * N2 + j) * N1 + i;
                p.x[3 * n] = -0.5 + (float)i / N1; // p.x[n][0]=x ..,y,z
                p.x[3 * n + 1] = -0.5 + (float)j / N2;
                p.x[3 * n + 2] = -0.5 + (float)k / N3;

                p.f[n][0] = sin(i) + 2 * sin(j) + 3 * sin(k) + 5 * ((i == 2) & (j == 4) & (k == 6));
                p.f[n][1] = 0.0;
            }
    nfftf_set_num_threads(8);
    cout <<"num threads = "<< nfftf_get_num_threads() << endl;

    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New(); // Create the vtkImageData object
    imageData->SetDimensions(N);                                                    // Set the dimensions of the image data
    imageData->SetSpacing(dd[0], dd[1], dd[2]);                                     // x,y,z
    imageData->SetOrigin(posl[0], posl[1], posl[2]);                                // Set the origin of the image data
    imageData->AllocateScalars(VTK_FLOAT, 3);
    imageData->GetPointData()->GetScalars()->SetName("f");
    float *data2 = static_cast<float *>(imageData->GetScalarPointer()); // Get a pointer to the density field array

    for (int k = 0; k < N3; ++k)
        for (int j = 0; j < N2; ++j)
            for (int i = 0; i < N1; ++i) // i is x-coord and array is [k][j][i]
            {
                int n = (k * N2 + j) * N1 + i;
                data2[3 * n + 2] = 0;
                for (int c = 0; c < 2; ++c)
                    data2[3 * n + c] = p.f[n][c];
            }

    vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New(); // Create the vtkXMLImageDataWriter object
    writer->SetFileName("f.vti");                                                                  // Set the output file name                                                                     // Set the time value
    writer->SetDataModeToBinary();
    writer->SetCompressorTypeToZLib(); // Enable compression
    writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
    writer->SetInputData(imageData);   // Set the input image data
    writer->Write();                   // Write the output file

    imageData->GetPointData()->GetScalars()->SetName("x");

    for (int k = 0; k < N3; ++k)
        for (int j = 0; j < N2; ++j)
            for (int i = 0; i < N1; ++i) // i is x-coord and array is [k][j][i]
            {
                int n = (k * N2 + j) * N1 + i;
                for (int c = 0; c < 3; ++c)
                    data2[3 * n + c] = p.x[3 * n + c]; // x,y,z
            }

    writer->SetFileName("x.vti"); // Set the output file name                                                                     // Set the time value
    writer->Write();              // Write the output file

    if (p.flags & PRE_ONE_PSI)
        nfftf_precompute_one_psi(&p);

    const char *check_error_msg = nfftf_check(&p);
    if (check_error_msg != 0)
    {
        printf("Invalid nfft parameter: %s\n", check_error_msg);
        return 1;
    }

    nfftf_adjoint(&p); /* Segfault if M <= 6 */

    imageData->GetPointData()->GetScalars()->SetName("f_hat");
    //  float *data2 = static_cast<float *>(imageData->GetScalarPointer()); // Get a pointer to the density field array

    for (int k = 0; k < N3; ++k)
    {
        for (int j = 0; j < N2; ++j)
        {
            for (int i = 0; i < N1; ++i)
            {
                int n = (k * N2 + j) * N1 + i;
                data2[3 * n + 2] = 0;
                for (int c = 0; c < 2; ++c)
                    data2[3 * n + c] = p.f_hat[n][c];
            }
        }
    }

    writer->SetFileName("f_hat.vti"); // Set the output file name                                                                     // Set the time value

    //  writer->SetInputData(imageData);   // Set the input image data
    writer->Write(); // Write the output file

    // nfftf_trafo(&p); /* Segfault if M <= 6 */
    fftwf_plan ifft = fftwf_plan_dft_3d(N3, N2, N2, p.f_hat, p.f, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Execute the inverse FFT
    cout << "execute the inverse FFTW plan" << endl;
    // nfftf_fftshift_complex(p.f_hat, 3, Nn);
    // fftwf_execute(ifft);
    //  nfftf_fftshift_complex(p.f, 3, Nn);

    nfftf_trafo(&p);

    imageData->GetPointData()->GetScalars()->SetName("f_new");
    //  float *data2 = static_cast<float *>(imageData->GetScalarPointer()); // Get a pointer to the density field array

    for (int k = 0; k < N3; ++k)
    {
        for (int j = 0; j < N2; ++j)
        {
            for (int i = 0; i < N1; ++i)
            {
                int n = (k * N2 + j) * N1 + i;
                int n1 = (i * N2 + j) * N3 + k;
                data2[3 * n + 2] = 0;
                for (int c = 0; c < 2; ++c)
                    data2[3 * n + c] = p.f[n][c];
            }
        }
    }

    writer->SetFileName("f_new.vti"); // Set the output file name                                                                     // Set the time value

    //  writer->SetInputData(imageData);   // Set the input image data
    writer->Write();    // Write the output file
                        //   nfftf_vpr_complex(p.f_hat, p.N_total, "adjoint nfft, vector f_hat");
    nfftf_finalize(&p); /* Segfault if M > 6 */
    return 0;
}