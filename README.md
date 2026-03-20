# FastGJK

FastGJK is a header-only library implementing CUDA accelerated GJK and EPA (TODO) algorithms.

# Build
```bash
cmake -B build
cmake --build build [-D<build_option>=<value>...]
```

The following table lists the build options.
|  Build option   | Default value  | Description |
|  ----  | ----  | ---- |
| `FASTGJK_BUILD_TESTS=ON/OFF`  | `ON` | Build unit tests |

# Acknowledgements
Our code is heavily borrowed from [OpenGJK](https://github.com/MattiaMontanari/openGJK).