#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "simulator_backend.h"

namespace py = pybind11;

py::dict run_cascade_dispatch(
    double p, double ell, int initial_intensity, bool record_history, int max_steps
) {
    CascadeMetricsResult result = run_cascade_cpp(p, ell, initial_intensity, record_history, max_steps);
    
    // Convert the C++ map to a Python dictionary for the history
    py::dict py_history;
    if (record_history) {
        for (const auto& pair : result.temporal_history) {
            py::dict generation;
            generation["nodes"] = py::int_(pair.second.size());
            generation["intensities"] = py::cast(pair.second);
            py_history[py::int_(pair.first)] = generation;
        }
    }

    py::dict payload;
    payload["size"] = py::int_(result.size);
    payload["depth"] = py::int_(result.depth);
    payload["max_width"] = py::int_(result.max_width);
    payload["total_intensity_effort"] = py::float_(result.total_intensity_effort);
    payload["temporal_history"] = record_history ? py::object(py_history) : py::none();
    return payload;
}

PYBIND11_MODULE(simulator_backend_cpp, m) {
    m.doc() = "High-performance C++ backend for SRC simulations";
    m.def("run_cascade_cpp", &run_cascade_dispatch, "Run a single SRC cascade with the standard dynamics",
          py::arg("p"), py::arg("ell"), py::arg("initial_intensity"),
          py::arg("record_history"), py::arg("max_steps"));
}
