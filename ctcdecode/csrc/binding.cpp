#include "path_trie.h"
#include "decoder_utils.h"
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>


void prune(std::vector<PathTrie *> &prefixes, int beam_size)
{
    if (prefixes.size() >= beam_size)
    {
        std::nth_element(prefixes.begin(),
                         prefixes.begin() + beam_size,
                         prefixes.end(),
                         prefix_compare);
        for (size_t i = beam_size; i < prefixes.size(); ++i)
        {
            prefixes[i]->remove();
        }
        prefixes.erase(prefixes.begin() + beam_size, prefixes.end());
    }
}

// PYBIND11_MAKE_OPAQUE(std::vector<std::pair<int, PathTrie *>>);
PYBIND11_MAKE_OPAQUE(std::vector<PathTrie *,std::allocator<PathTrie*> >);

// PYBIND11_MAKE_OPAQUE(std::vector<std::pair<int, PathTrie *>>)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // py::bind_vector<std::vector<std::pair<int, PathTrie *>> >(m, "TrieTuple");
    py::bind_vector<std::vector<PathTrie *,std::allocator<PathTrie*>>>(m, "ListTrie");
    py::class_<PathTrie>(m, "PathTrie")
        .def(py::init<>())
        .def("get_path_trie", &PathTrie::get_path_trie, py::return_value_policy::reference)
        .def("get_path_vec", (PathTrie * (PathTrie::*)(std::vector<int> &, std::vector<int> &)) & PathTrie::get_path_vec, py::return_value_policy::reference)
        .def("iterate_to_vec", &PathTrie::iterate_to_vec)
        .def("remove", &PathTrie::remove)
        .def_readwrite("p_b", &PathTrie::log_prob_b_prev)
        .def_readwrite("p_nb", &PathTrie::log_prob_nb_prev)
        .def_readwrite("n_p_b", &PathTrie::log_prob_b_cur)
        .def_readwrite("n_p_nb", &PathTrie::log_prob_nb_cur)
        .def_readwrite("score", &PathTrie::score)
        .def_readwrite("score_ctc", &PathTrie::score_ctc)
        .def_readwrite("score_lm", &PathTrie::score_lm)
        .def_readwrite("character", &PathTrie::character)
        .def_readwrite("timestep", &PathTrie::timestep)
        .def_readonly("parent", &PathTrie::parent, py::return_value_policy::reference)
        .def_property_readonly("prefix", [](PathTrie &self) {
            std::vector<int> prefix;
            std::vector<int> timesteps;
            self.get_path_vec(prefix, timesteps);
            return prefix;
        }, py::return_value_policy::copy)
        .def_property_readonly("timesteps", [](PathTrie &self) {
            std::vector<int> prefix;
            std::vector<int> timesteps;
            self.get_path_vec(prefix, timesteps);
            return timesteps;
        }, py::return_value_policy::copy)
        .def_property_readonly("children", &PathTrie::getChildren, py::return_value_policy::reference)
        .def("__repr__", [](const PathTrie &self){
            return "Node(key=" + std::to_string(self.character) + ", timestep=" +  std::to_string(self.timestep) + ", p_b=" + std::to_string(self.log_prob_b_prev) + ", p_nb=" + std::to_string(self.log_prob_nb_prev) + "score=" + std::to_string(self.score) + ")";
        });
    m.def("prune", &prune);
}