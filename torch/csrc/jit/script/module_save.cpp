#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/export.h>

namespace torch {
namespace jit {
namespace script {

void Module::save(std::ostream& out, const ExtraFilesMap& extra_files) const {
  ExportModule(*this, out, extra_files, false);
}

void Module::save(const std::string& filename, const ExtraFilesMap& extra_files)
    const {
  ExportModule(*this, filename, extra_files, false);
}

void Module::_save_for_mobile(
    std::ostream& out,
    const ExtraFilesMap& extra_files) const {
  ExportModule(*this, out, extra_files, true);
}

void Module::_save_for_mobile(
    const std::string& filename,
    const ExtraFilesMap& extra_files) const {
  ExportModule(*this, filename, extra_files, true);
}

} // namespace script
} // namespace jit
} // namespace torch
