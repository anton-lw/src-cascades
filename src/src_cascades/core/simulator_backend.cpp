#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "simulator_backend.h"

namespace py = pybind11;

// Wrapper for the templated C++ simulator function.
// It accepts Python callables for the dynamics and termination conditions.
py::dict run_cascade_dispatch(
    double p, double ell, int initial_intensity, bool record_history, int max_steps,
    const std::function<int(int, bool)>& dynamics_func,
    const std::function<bool(int)>& termination_func
) {
    CascadeMetricsResult result = run_cascade_cpp(p, ell, initial_intensity, record_history, max_steps, dynamics_func, termination_func);
    
    // Convert the C++ map to a Python dictionary for the history
    py::dict py_history;
    if (record_history) {
        for (const auto& pair : result.temporal_history) {
            py_history[py::int_(pair.first)] = py::dict(
                py::arg("nodes") = pair.second.size(),
                py::arg("intensities") = py::cast(pair.second)
            );
        }
    }

    return py::dict(
        py::arg("size") = result.size,
        py::arg("depth") = result.depth,
        py::arg("max_width") = result.max_width,
        py::arg("total_intensity_effort") = result.total_intensity_effort,
        py::arg("temporal_history") = record_history ? py_history : py::none()
    );
}

PYBIND11_MODULE(simulator_backend_cpp, m) {
    m.doc() = "High-performance C++ backend for SRC simulations";
    m.def("run_cascade_cpp", &run_cascade_dispatch, "Run a single SRC cascade with custom dynamic and termination functions",
          py::arg("p"), py::arg("ell"), py::arg("initial_intensity"),
          py::arg("record_history"), py::arg("max_steps"), 
          py::arg("dynamics_func"), py::arg("termination_func"));
}