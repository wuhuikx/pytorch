#include <torch/jit.h>
#include <torch/script.h>
#include <torch/csrc/jit/script/module.h>

#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[]) {
  std::string input_file_path{argv[1]};
  std::string output_file_path{argv[2]};

  std::ifstream ifs(input_file_path);
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  torch::jit::script::Module m("TestModule");

  m.define(buffer.str());
  m.save(output_file_path);
}
