#include "common.cu"
#include "draw.cu"
#include "display.cu"

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    // create resources on the video card

    float *hummel;
    glm::vec4 *canvas;

    // load hummel.png
    {
        float hummel_data[Size];

        int w, h;
        unsigned char* image_data = SOIL_load_image("ichbineinehummel.png", &w, &h, 0, SOIL_LOAD_RGBA);
        assert(image_data);
        assert(w == WinSize.x && h == WinSize.y);

        auto gray = [] (glm::vec3 c) { return (c.r+c.b+c.g*2.0f)/4.0f; };

        // read data and flip y
        for (int y = 0; y < WinSize.y; ++y) {
            for (int x = 0; x < WinSize.x; ++x) {
                int i = x + y*WinSize.x;
                int j = x + Size - y*WinSize.x - 1;
                auto c = glm::vec3(image_data[j*4], image_data[j*4+1], image_data[j*4+2]) / 255.0f;
                hummel_data[i] = step(0.27f, 1.0f - gray(c));
            }
        }

        SOIL_free_image_data(image_data);

        cudaMallocManaged(&hummel, Size * sizeof(float));
        cudaMemcpy(hummel, hummel_data, Size * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMallocManaged(&canvas, Size * sizeof(glm::vec4));
    cudaMemset(&canvas, 0, Size * sizeof(glm::vec4));

    cudaDeviceSynchronize();

    // render an canvas using ray marching
    auto update = [=] () {
        cudaDeviceSynchronize();
        draw(canvas, hummel);

        cudaDeviceSynchronize();
        return canvas;
    };

    // main loop. render and display
    display(update);

    // clean up resources
    cudaFree(hummel);
    cudaFree(canvas);

    return 0;
}
