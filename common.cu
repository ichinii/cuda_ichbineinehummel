#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <memory>
#include <map>
#include <numeric>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <SOIL/SOIL.h>

using namespace glm;

template <typename T>
constexpr T integralCeilDivision(T a, T b) {
    return ((a-1)/b)+1;
}

constexpr const auto WinSize = ivec2(512, 512);
static_assert(WinSize.x == WinSize.y);
constexpr const auto Size = WinSize.x * WinSize.y;
constexpr float WinDiag = WinSize.x * 1.41421356237309504880f;

constexpr const auto W = 256;
constexpr const auto B = integralCeilDivision(Size, W);

__device__ float n21(vec2 s) {
    return fract(12095.283 * sin(dot(vec2(585.905, 821.895), s)));
}

__device__ float n31(vec3 s) {
    return fract(9457.824 * sin(dot(vec3(385.291, 458.958, 941.950), s)));
}

__device__ vec3 n33(vec3 s) {
    float n1 = n21(vec2(s));
    float n2 = n21(vec2(s.z, s.x));
    float n3 = n21(vec2(s.y, s.z));
    return vec3(n1, n2, n3);
}
