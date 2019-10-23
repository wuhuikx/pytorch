
require 'xcodeproj'
require 'fileutils'

puts "Current directory: #{Dir.pwd}"

install_path = File.expand_path("../../../build_ios/install")
if not Dir.exist? (install_path) 
    raise "path doesn't exist:#{install_path}!"
end
xcodeproj_path = File.expand_path("../TestApp.xcodeproj")
if not File.exist? (xcodeproj_path) 
    raise "path doesn't exist:#{xcodeproj_path}!"
end

puts "Setting up TestApp.xcodeproj..."
project = Xcodeproj::Project.open(xcodeproj_path)
target = project.targets.first #TestApp
header_search_path      = ['$(inherited)', "#{install_path}/include"]
libraries_search_path   = ['$(inherited)', "#{install_path}/lib"]
other_linker_flags      = ['$(inherited)', "-all_load"]

target.build_configurations.each do |config|
    config.build_settings['HEADER_SEARCH_PATHS']    = header_search_path
    config.build_settings['LIBRARY_SEARCH_PATHS']   = libraries_search_path
    config.build_settings['OTHER_LDFLAGS']          = other_linker_flags
    config.build_settings['ENABLE_BITCODE']         = 'No'
end

puts "Installing the testing model..."
model_path = File.expand_path("./model.pt")
if not File.exist?(model_path)
   raise "no model can be found!"
end
group = project.main_group.find_subpath(File.join('TestApp'),true)
group.set_source_tree('SOURCE_ROOT')
group.files.each do |file|
    if file.name.to_s.end_with?(".pt")
        puts "Found old model, remove it"
        group.remove_reference(file)
        target.resources_build_phase.remove_file_reference(file)
    end
end
model_file_ref = group.new_reference(model_path)
target.resources_build_phase.add_file_reference(model_file_ref, true)

puts "Linking static libraries..."
target.frameworks_build_phases.clear
libs = ['libc10.a', 'libclog.a', 'libnnpack.a', 'libeigen_blas.a', 'libcpuinfo.a', 'libpytorch_qnnpack.a', 'libtorch.a']
for lib in libs do 
    path = "#{install_path}/lib/#{lib}"
    if File.exist?(path)
        libref = project.frameworks_group.new_file(path)
        target.frameworks_build_phases.add_file_reference(libref)
    end
end
project.save
puts "Done."
